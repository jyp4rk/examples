# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import configparser
import copy
import os
import time
from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from FasterTransformer.examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT  # type: ignore

from scripts.inference.convert_hf_mpt_to_ft import convert_mpt_to_ft  # isort: skip # type: ignore

LOCAL_CHECKPOINT_PATH = '/tmp/mpt'

# Data set used by locust benchmark
EXAMPLE_INPUTS_TEXT = [
    # Short queries with short responses.
    "Finish this story: I awoke with a _",
    "How many atoms combine to form dioxygen?",
    # Short queries which typically produce medium-to-long answers.
    "My name is ",
    "What is 1+1?",
    "What is a lion?",
    "Raffi:\nHave you ever seen a whale with a polka dot tail?\n You:",
    "Write a short story about a robot that has a nice day.",
    "What are some of the most common misconceptions about birds?",
    # Long query with typically long answers (from SQuAD)
    "The immune system is a system of many biological structures and processes within an organism that protects against disease. To function properly, an immune system must detect a wide variety of agents, known as pathogens, from viruses to parasitic worms, and distinguish them from the organism's own healthy tissue. In many species, the immune system can be classified into subsystems, such as the innate immune system versus the adaptive immune system, or humoral immunity versus cell-mediated immunity. In humans, the blood–brain barrier, blood–cerebrospinal fluid barrier, and similar fluid–brain barriers separate the peripheral immune system from the neuroimmune system which protects the brain.\nWhat is the immune system?",
]


class MPTFTModelHandler():

    DEFAULT_GENERATE_KWARGS = {
        # Output sequence length to generate.
        'max_length': 256,
        # Beam width for beam search
        'beam_width': 1,
        # top k candidate number
        'top_k': 0,
        # top p probability threshold
        'top_p': 0.95,
        # temperature parameter
        'temperature': 0.8,
        # Penalty for repetitions
        'repetition_penalty': 1.0,
        # Presence penalty. Similar to repetition, but additive rather than multiplicative.
        'presence_penalty': 0.0,
        'beam_search_diversity_rate': 0.0,
        'len_penalty': 0.0,
        'bad_words_list': None,
        # A minimum number of tokens to generate.
        'min_length': 0,
        # if True, use different random seed for sentences in a batch.
        'random_seed': True
    }

    INPUT_STRINGS_KEY = 'input_strings'

    def __init__(self,
                 model_name: str,
                 ft_lib_path: str,
                 inference_data_type: str = 'bf16',
                 int8_mode: int = 0,
                 gpus: int = 1):
        """Fastertransformer model handler for MPT foundation series.

        Args:
            model_name (str): Name of the model as on HF hub (e.g., mosaicml/mpt-7b-instruct).
            ft_lib_path (str): Path to the libth_transformer dynamic lib file(.e.g., build/lib/libth_transformer.so).
            inference_data_type (str): Data type to use for inference (Default: bf16)
            int8_mode (int): The level of quantization to perform. 0: No quantization. All computation in data_type,
                1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32
            gpus (int): Number of gpus to use for inference (Default: 1)
        """
        self.device = torch.cuda.current_device()
        self.model_name = model_name

        # Datatype of weights in the HF checkpoint
        weight_data_type = 'fp32'
        model_path = os.path.join(LOCAL_CHECKPOINT_PATH, f'{gpus}-gpu')
        ckpt_config_path = os.path.join(model_path, 'config.ini')
        # Reuse checkpoint if it exists.
        if not os.path.isfile(ckpt_config_path):
            convert_mpt_to_ft(self.model_name, LOCAL_CHECKPOINT_PATH, gpus,
                              weight_data_type, False)
        else:
            print(f'Reusing existing FT checkpoint at {model_path}')
        ckpt_config = configparser.ConfigParser()
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)

        # Disable this optimization.
        # https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#advanced-features
        shared_contexts_ratio = 0.0

        if 'gpt' in ckpt_config.keys():
            head_num = ckpt_config.getint('gpt', 'head_num')
            size_per_head = ckpt_config.getint('gpt', 'size_per_head')
            vocab_size = ckpt_config.getint('gpt', 'vocab_size')
            start_id = ckpt_config.getint('gpt', 'start_id')
            end_id = ckpt_config.getint('gpt', 'end_id')
            layer_num = ckpt_config.getint('gpt', 'num_layer')
            max_seq_len = ckpt_config.getint('gpt', 'max_pos_seq_len')
            weights_data_type = ckpt_config.get('gpt', 'weight_data_type')
            tensor_para_size = ckpt_config.getint('gpt', 'tensor_para_size')
            pipeline_para_size = ckpt_config.getint('gpt',
                                                    'pipeline_para_size',
                                                    fallback=1)
            layernorm_eps = ckpt_config.getfloat('gpt',
                                                 'layernorm_eps',
                                                 fallback=1e-5)
            use_attention_linear_bias = ckpt_config.getboolean(
                'gpt', 'use_attention_linear_bias')
            has_positional_encoding = ckpt_config.getboolean(
                'gpt', 'has_positional_encoding')
        else:
            raise RuntimeError(
                'Unexpected config.ini for the FT checkpoint. Expected FT checkpoint to contain the `gpt` key.'
            )

        self.end_id = end_id

        print('Initializing ParallelGPT')
        self.model = ParallelGPT(
            head_num,
            size_per_head,
            vocab_size,
            start_id,
            end_id,
            layer_num,
            max_seq_len,
            tensor_para_size,
            pipeline_para_size,
            lib_path=ft_lib_path,
            inference_data_type=inference_data_type,
            int8_mode=int8_mode,
            weights_data_type=weights_data_type,
            layernorm_eps=layernorm_eps,
            use_attention_linear_bias=use_attention_linear_bias,
            has_positional_encoding=has_positional_encoding,
            shared_contexts_ratio=shared_contexts_ratio)
        print(f'Loading model from {model_path}')
        if not self.model.load(ckpt_path=model_path):
            raise RuntimeError(
                'Could not load model from a FasterTransformer checkpoint')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       trust_remote_code=True)
        print('FT initialization complete')

    def _parse_inputs(self, inputs: Dict[str, Any]):
        if self.INPUT_STRINGS_KEY not in inputs or not isinstance(
                inputs[self.INPUT_STRINGS_KEY], list):
            raise RuntimeError(
                'Input strings must be provided as a list to generate call')

        generate_input = inputs[self.INPUT_STRINGS_KEY]

        # Set default generate kwargs
        generate_kwargs = copy.deepcopy(self.DEFAULT_GENERATE_KWARGS)
        # If request contains any additional kwargs, add them to generate_kwargs
        for k, v in inputs.items():
            if k not in [self.INPUT_STRINGS_KEY]:
                generate_kwargs[k] = v

        return generate_input, generate_kwargs

    def _convert_kwargs(self, generate_input: List[str],
                        generate_kwargs: Dict[str, Any]):
        batch_size = len(generate_input)

        # Convert expected kwargs into types expected by FT.
        # Integer args may be floats if the values are from a json payload.
        generate_kwargs['max_length'] = int(generate_kwargs['max_length'])
        generate_kwargs['top_k'] = int(generate_kwargs['top_k']) * torch.ones(
            batch_size, dtype=torch.int32)
        generate_kwargs['top_p'] *= torch.ones(batch_size, dtype=torch.float32)
        generate_kwargs['temperature'] *= torch.ones(batch_size,
                                                     dtype=torch.float32)
        repetition_penalty = generate_kwargs['repetition_penalty']
        generate_kwargs[
            'repetition_penalty'] = None if repetition_penalty == 1.0 else repetition_penalty * torch.ones(
                batch_size, dtype=torch.float32)
        presence_penalty = generate_kwargs['presence_penalty']
        generate_kwargs[
            'presence_penalty'] = None if presence_penalty == 0.0 else presence_penalty * torch.ones(
                batch_size, dtype=torch.float32)
        generate_kwargs['beam_search_diversity_rate'] *= torch.ones(
            batch_size, dtype=torch.float32)
        generate_kwargs['len_penalty'] *= torch.ones(size=[batch_size],
                                                     dtype=torch.float32)
        generate_kwargs['min_length'] = int(
            generate_kwargs['min_length']) * torch.ones(size=[batch_size],
                                                        dtype=torch.int32)
        if generate_kwargs['random_seed']:
            generate_kwargs['random_seed'] = torch.randint(0,
                                                           10000,
                                                           size=[batch_size],
                                                           dtype=torch.int64)

    def predict(self, input_dicts: List[Dict[str, Any]]):
        generate_inputs = []
        generate_kwargs = {}
        for input_dict in input_dicts:
            generate_input_list, generate_kwarg = self._parse_inputs(input_dict)
            # Flatten the list of inputs into a single list
            # 2 cases for batching
            # 1. Multiple requests single input string
            # 2. Single request multiple input strings
            generate_inputs += generate_input_list

            for k, v in generate_kwarg.items():
                if k in generate_kwargs and generate_kwargs[k] != v:
                    raise RuntimeError(
                        f'Request has conflicting values for kwarg {k}')
                generate_kwargs[k] = v

        # Convert kwargs into torch types and populate values like 'random_seed'
        self._convert_kwargs(generate_inputs, generate_kwargs)

        start_ids = [
            torch.tensor(self.tokenizer.encode(c),
                         dtype=torch.int32,
                         device=self.device) for c in generate_inputs
        ]
        start_lengths = [len(ids) for ids in start_ids]
        start_ids = pad_sequence(start_ids,
                                 batch_first=True,
                                 padding_value=self.end_id)
        start_lengths = torch.IntTensor(start_lengths)
        tokens_batch = self.model(start_ids, start_lengths, **generate_kwargs)
        outputs = []
        for tokens in tokens_batch:
            for beam_id in range(generate_kwargs['beam_width']):
                # Do not exclude context input from the output
                # token = tokens[beam_id][start_lengths[i]:]
                token = tokens[beam_id]
                # stop at end_id; This is the same as eos_token_id
                token = token[token != self.end_id]
                output = self.tokenizer.decode(token)
                outputs.append(output)
        return outputs

    def predict_stream(self, **inputs: Dict[str, Any]):
        raise RuntimeError('Streaming is not supported with FasterTransformer!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--ft_lib_path',
        type=str,
        required=True,
        help=
        'Path to the libth_transformer dynamic lib file(e.g., build/lib/libth_transformer.so.'
    )
    parser.add_argument(
        '--name_or_dir',
        '-i',
        type=str,
        help=
        'HF hub Model name (e.g., mosaicml/mpt-7b) or local dir path to load checkpoint from',
        required=True)
    parser.add_argument('--inference_data_type',
                        '--data_type',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default='bf16')
    parser.add_argument(
        '--int8_mode',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'The level of quantization to perform. 0: No quantization. All computation in data_type. 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32'
    )
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='The number of gpus to use for inference.')
    parser.add_argument('--iterations',
                        type=int,
                        default=25,
                        help='The number of predict iterations to run.')

    args = parser.parse_args()

    model_handler = MPTFTModelHandler(args.name_or_dir, args.ft_lib_path,
                                      args.inference_data_type, args.int8_mode,
                                      args.gpus)

    times = []
    t_start = time.time()
    while len(times) < args.iterations:
        for ex in EXAMPLE_INPUTS_TEXT:
            if len(times) >= args.iterations:
                break
            inputs = {'input_strings': [ex]}
            t_pred_start = time.time()
            out = model_handler.predict([inputs])
            t_pred = time.time() - t_pred_start
            print(f'{out[0]} | t: {t_pred}')
            t_pred = int(t_pred * 100) * 10  # ~nearest 10 ms
            times.append(t_pred)
    t_total = time.time() - t_start
    n = len(times)
    print(f'rate: {n/t_total} predict/sec | time: {t_total} sec')
    times.sort()
    print(times)
    print(
        f'p50: {times[int(n/2)]} ms | p75: {times[int(0.75*n)]} ms | p90: {times[int(0.9*n)]} | max: {times[-1]}'
    )
