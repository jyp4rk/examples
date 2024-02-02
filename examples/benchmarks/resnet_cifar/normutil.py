# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
# from MinkowskiEngine import SparseTensor

# class MinkowskiGRN(nn.Module):
#     """ GRN layer for sparse tensors.
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, dim))

#     def forward(self, x):
#         cm = x.coordinate_manager
#         in_key = x.coordinate_map_key

#         Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return SparseTensor(
#                 self.gamma * (x.F * Nx) + self.beta + x.F,
#                 coordinate_map_key=in_key,
#                 coordinate_manager=cm)

# class MinkowskiDropPath(nn.Module):
#     """ Drop Path for sparse tensors.
#     """

#     def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
#         super(MinkowskiDropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep

#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         cm = x.coordinate_manager
#         in_key = x.coordinate_map_key
#         keep_prob = 1 - self.drop_prob
#         mask = torch.cat([
#             torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
#             else torch.zeros(len(_)) for _ in x.decomposed_coordinates
#         ]).view(-1, 1).to(x.device)
#         if keep_prob > 0.0 and self.scale_by_keep:
#             mask.div_(keep_prob)
#         return SparseTensor(
#                 x.F * mask,
#                 coordinate_map_key=in_key,
#                 coordinate_manager=cm)

# class MinkowskiLayerNorm(nn.Module):
#     """ Channel-wise layer normalization for sparse tensors.
#     """

#     def __init__(
#         self,
#         normalized_shape,
#         eps=1e-6,
#     ):
#         super(MinkowskiLayerNorm, self).__init__()
#         self.ln = nn.LayerNorm(normalized_shape, eps=eps)
#     def forward(self, input):
#         output = self.ln(input.F)
#         return SparseTensor(
#             output,
#             coordinate_map_key=input.coordinate_map_key,
#             coordinate_manager=input.coordinate_manager)

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", affine=True):
        super().__init__()
        if affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            weight = torch.ones(normalized_shape)
            bias = torch.zeros(normalized_shape)
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class BatchNorm2d_only_variance(nn.Module):
    """ BatchNorm2d that only uses the variance of the input for normalization.
    """
    def __init__(self, num_features, eps=1e-6, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            var = x.pow(2).mean(dim=(0, 2, 3), keepdim=True)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            var = self.running_var
            x = x / torch.sqrt(var + self.eps)[None, :, None, None]
        # x = x / torch.sqrt(var + self.eps)
        if self.affine:
            x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x

# class _NormBase(Module):
#     """Common base of _InstanceNorm and _BatchNorm."""

#     _version = 2
#     __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
#     num_features: int
#     eps: float
#     momentum: float
#     affine: bool
#     track_running_stats: bool
#     # WARNING: weight and bias purposely not defined here.
#     # See https://github.com/pytorch/pytorch/issues/39670

#     def __init__(
#         self,
#         num_features: int,
#         eps: float = 1e-5,
#         momentum: float = 0.1,
#         affine: bool = True,
#         track_running_stats: bool = True,
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
#             self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
#             self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
#             self.running_mean: Optional[Tensor]
#             self.running_var: Optional[Tensor]
#             self.register_buffer('num_batches_tracked',
#                                  torch.tensor(0, dtype=torch.long,
#                                               **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
#             self.num_batches_tracked: Optional[Tensor]
#         else:
#             self.register_buffer("running_mean", None)
#             self.register_buffer("running_var", None)
#             self.register_buffer("num_batches_tracked", None)
#         self.reset_parameters()

#     def reset_running_stats(self) -> None:
#         if self.track_running_stats:
#             # running_mean/running_var/num_batches... are registered at runtime depending
#             # if self.track_running_stats is on
#             self.running_mean.zero_()  # type: ignore[union-attr]
#             self.running_var.fill_(1)  # type: ignore[union-attr]
#             self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

#     def reset_parameters(self) -> None:
#         self.reset_running_stats()
#         if self.affine:
#             init.ones_(self.weight)
#             init.zeros_(self.bias)

#     def _check_input_dim(self, input):
#         raise NotImplementedError

#     def extra_repr(self):
#         return (
#             "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
#             "track_running_stats={track_running_stats}".format(**self.__dict__)
#         )

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         version = local_metadata.get("version", None)

#         if (version is None or version < 2) and self.track_running_stats:
#             # at version 2: added num_batches_tracked buffer
#             #               this should have a default value of 0
#             num_batches_tracked_key = prefix + "num_batches_tracked"
#             if num_batches_tracked_key not in state_dict:
#                 state_dict[num_batches_tracked_key] = (
#                     self.num_batches_tracked
#                     if self.num_batches_tracked is not None and self.num_batches_tracked.device != torch.device('meta')
#                     else torch.tensor(0, dtype=torch.long)
#                 )

#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )


# class _BatchNorm(_NormBase):
#     def __init__(
#         self,
#         num_features: int,
#         eps: float = 1e-5,
#         momentum: float = 0.1,
#         affine: bool = True,
#         track_running_stats: bool = True,
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__(
#             num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
#         )

#     def forward(self, input: Tensor) -> Tensor:
#         self._check_input_dim(input)

#         # exponential_average_factor is set to self.momentum
#         # (when it is available) only so that it gets updated
#         # in ONNX graph when this node is exported to ONNX.
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum

#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:  # type: ignore[has-type]
#                 self.num_batches_tracked.add_(1)  # type: ignore[has-type]
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         r"""
#         Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#         Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#         """
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)

#         r"""
#         Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
#         passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
#         used for normalization (i.e. in eval mode when buffers are not None).
#         """
#         return F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean
#             if not self.training or self.track_running_stats
#             else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             self.weight,
#             self.bias,
#             bn_training,
#             exponential_average_factor,
#             self.eps,
#         )
