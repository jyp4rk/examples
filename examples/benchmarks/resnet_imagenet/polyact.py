import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from torch import Tensor
from normutil import *


class Quad(nn.Module):
    def __init__(
        self,
        bias: Tensor,
        weight1: Tensor,
        weight2: Tensor
    ):
        super(Quad, self).__init__()  # init the base class
        self.b = bias.reshape(1, -1, 1, 1)
        self.weight1 = weight1.reshape(1, -1, 1, 1)
        self.weight2 = weight2.reshape(1, -1, 1, 1)
        self.register_parameter('bias', nn.Parameter(self.b))
        # self.register_parameter('weight1', nn.Parameter(self.weight1))
        # self.register_parameter('weight2', nn.Parameter(self.weight2))

    def get_weights(self):
        return self.bias, self.weight1, self.weight2

    def forward(self, inputs):
        return self.weight2 * inputs**2 + self.weight1 * inputs + self.bias


class aespa(nn.Module):
    def __init__(
        self,
        num_pol: int,
        planes: int,
        lr_scaler: float = 1.0,
        decay=0.99,
        reuse=False,
        norm: str = 'bn',
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        '''
        Init method.
        '''
        self.h = []
        self.num_pol = num_pol
        self.lr_scaler = lr_scaler
        self.decay = decay
        self.reuse = reuse
        self.planes = planes
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.track_running_stats = track_running_stats
        super().__init__()  # init the base class

        # hermite polynomials

        def h0(x): return torch.ones_like(x)
        self.h.append(h0)

        def h1(x): return x
        self.h.append(h1)

        def h2(x): return (x**2 - 1)/np.sqrt(np.math.factorial(2))
        self.h.append(h2)

        def h3(x): return (x**3 - 3*x)/np.sqrt(np.math.factorial(3))
        self.h.append(h3)

        def h4(x): return (x**4 - 6*(x**2) + 3)/np.sqrt(np.math.factorial(4))
        self.h.append(h4)

        def h5(x): return (x**5 - 10*x**3 + 15*x)/np.sqrt(np.math.factorial(5))
        self.h.append(h5)

        def h6(x): return (x**6 - 15*x**4 + 45*x**2 - 15) / \
            np.sqrt(np.math.factorial(6))
        self.h.append(h6)

        def h7(x): return (x**7 - 21*x**5 + 105*x**3 - 105*x) / \
            np.sqrt(np.math.factorial(7))
        self.h.append(h7)

        def h8(x): return (x**8 - 28*x**6 + 210*x**4 -
                           420*x**2 + 105)/np.sqrt(np.math.factorial(8))
        self.h.append(h8)

        def h9(x): return (x**9 - 36*x**7 + 378*x**5 - 1260 *
                           x**3 + 945*x)/np.sqrt(np.math.factorial(9))
        self.h.append(h9)

        def h10(x): return (x**10 - 45*x**8 + 630*x**6 - 3150*x **
                            4 + 4725*x**2 - 945)/np.sqrt(np.math.factorial(10))

        self.h.append(h10)

        # coefficients for hermite polynomials of ReLU
        self.w_0 = torch.Tensor([1.0/np.sqrt(2*np.pi)])
        self.w_1 = torch.Tensor([1.0/2])
        self.w_2 = torch.Tensor([1.0/np.sqrt(4*np.pi)])

        self.norm = norm

        ## activation function-wise trainable coefficients
        ## normalize both the first and second degree terms
        if norm == 'bn':
            self.norm1 = nn.BatchNorm2d(
                self.planes, affine=False, track_running_stats=True, **factory_kwargs)
            self.norm2 = nn.BatchNorm2d(
                self.planes, affine=False, track_running_stats=True, **factory_kwargs)

        ## activation function-wise trainable coefficients
        ## normalize only for the second degree term
        elif norm == 'bnv2':
            self.norm1 = nn.Identity()
            self.norm2 = nn.BatchNorm2d(
                self.planes,affine=False, **factory_kwargs)
            self.w_1 = torch.Tensor([1.0])

        ## basis-wise trainable coefficients with lr_scaler
        elif norm == 'bnv3':
            self.norm1 = nn.BatchNorm2d(
                self.planes,affine=False, **factory_kwargs)
            self.norm2 = nn.BatchNorm2d(
                self.planes,affine=False, **factory_kwargs)
        ## LayerNorm from ConvNext
        ## https://github.com/facebookresearch/ConvNeXt
        elif norm == 'ln':
            self.norm1 = LayerNorm(self.planes, eps=1e-6, affine=False)
            self.norm2 = LayerNorm(self.planes, eps=1e-6, affine=False)
        else:
            raise NotImplementedError(
                'norm layer {} is not implemented'.format(norm))

        if norm in ['bn', 'bnv2']:
            self.register_buffer(
                'w0', (self.w_0))
            self.register_buffer(
                'w1', (self.w_1))
            self.register_buffer(
                'w2', (self.w_2))
            weight = nn.Parameter(torch.ones(self.planes, **factory_kwargs))
            bias = nn.Parameter(torch.zeros(self.planes, **factory_kwargs))
            self.register_parameter(
                'herpn_weight', weight)
            self.register_parameter(
                'herpn_bias', bias)
            self.herpn_weight: Optional[Tensor]
            self.herpn_bias: Optional[Tensor]
        else:
            self.wp0 = nn.Parameter(self.w_0)
            self.wp1 = nn.Parameter(self.w_1)
            self.wp2 = nn.Parameter(self.w_2)
            if self.lr_scaler is None:
                raise ValueError('lr scaler is not defined')

    def aespa(self, inp: torch.Tensor) -> torch.Tensor:

        if self.norm == 'bn':
            xx =  self.w0 * self.h[0](inp) + self.w1 * self.h[1](self.norm1(inp)) + self.w2 * self.norm2(self.h[2](inp))
        elif self.norm == 'bnv2':
            xx =  self.w0 * self.h[0](inp) + self.w1 * self.h[1](self.norm1(inp)) + self.w2 * self.norm2(self.h[2](inp))
        elif self.norm == 'bnv3':
            w0 = (self.wp0 - self.wp0 * self.lr_scaler).detach() + self.wp0 * self.lr_scaler
            w1 = (self.wp1 - self.wp1 * self.lr_scaler).detach() + self.wp1 * self.lr_scaler
            w2 = (self.wp2 - self.wp2 * self.lr_scaler).detach() + self.wp2 * self.lr_scaler
            xx =  w0 * self.h[0](inp) + w1 * self.h[1](self.norm1(inp)) + w2 * self.norm2(self.h[2](inp))
            return xx
        return self.herpn_weight[None, :, None, None] * xx + self.herpn_bias[None, :, None, None]

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'fused_hermite'):
            return self.fused_hermite(inp)
        else:
            return self.aespa(inp)
    def get_equivalent_quad_activation(self):
        return self._fuse_hermite_bn()
    def _fuse_hermite_bn(self):
        rstd = torch.rsqrt(self.running_var+1e-8)
        bn_bias0, bn_bias1, bn_bias2 = torch.chunk(self.running_mean,3,0)
        rstd_bias, rstd_weight1, rstd_weight2 = torch.chunk(rstd,3,0)
        ## AEPSA has no gamma (w), beta (b) for BN
        bn_bias = self.f_0*rstd_bias*(1-bn_bias0) - self.f_1*rstd_weight1*bn_bias1 - self.f_2*rstd_weight2*bn_bias2
        hermite_bias = self.herpn_bias + self.herpn_weight * bn_bias
        hermite_weight1 = self.f_1*rstd_weight1 * self.herpn_weight
        hermite_weight2 = self.f_2*rstd_weight2 * self.herpn_weight /np.sqrt(np.math.factorial(2))
        hermite_bias -= hermite_weight2
        return hermite_bias, hermite_weight1, hermite_weight2

    def switch_to_deploy(self):
        hermite_bias, hermite_weight1, hermite_weight2 = self.get_equivalent_quad_activation()
        self.fused_hermite = Quad(hermite_bias, hermite_weight1,hermite_weight2)
        # for param in self.parameters():
        #     param.detach_()
        # self.register_parameter('fused_hermite_bias',torch.nn.Parameter(self.fused_hermite.bias))
        # self.register_parameter('fused_hermite_weight1',torch.nn.Parameter(self.fused_hermite.weight1))
        # self.register_parameter('fused_hermite_weight2',torch.nn.Parameter(self.fused_hermite.weight2))
        self.deploy=True



class PolyAct(nn.Module):
    def __init__(self, trainable=False, init_coef=None, in_scale=1,
                 out_scale=1,
                 train_scale=False):
        super(PolyAct, self).__init__()
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393, 0.0]
        self.deg = len(init_coef) - 1
        self.trainable = trainable
        coef = torch.Tensor(init_coef)

        if trainable:
            self.coef = nn.Parameter(coef, requires_grad=True)
        else:
            self.register_buffer('coef', coef)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None


    def forward(self, x):
        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        return x

    def __repr__(self):
        print_coef = self.coef.cpu().detach().numpy()
        return "PolyAct(trainable={}, coef={})".format(
            self.trainable, print_coef)

    def calc_polynomial(self, x):
        res = self.coef[0] + self.coef[1] * x
        for i in range(2, self.deg):
            res = res + self.coef[i] * (x ** i)

        return res


class PolyActPerChannel(nn.Module):
    def __init__(self, channels, init_coef=None, data_format="channels_first", in_scale=1,
                 out_scale=1,
                 train_scale=False):
        super(PolyActPerChannel, self).__init__()
        self.channels = channels
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        self.deg = len(init_coef) - 1
        coef = torch.Tensor(init_coef)
        coef = coef.repeat([channels, 1])
        coef = torch.unsqueeze(torch.unsqueeze(coef, -1), -1)
        self.coef = nn.Parameter(coef, requires_grad=True)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None

        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        if self.data_format == 'channels_last':
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        return x

    def __repr__(self):
        # print_coef = self.coef.cpu().detach().numpy()
        print_in_scale = self.in_scale.cpu().detach().numpy() if self.in_scale else None
        print_out_scale = self.out_scale.cpu().detach().numpy() if self.out_scale else None

        return "PolyActPerChannel(channels={}, in_scale={}, out_scale={})".format(
            self.channels, print_in_scale, print_out_scale)

    def calc_polynomial(self, x):

        if self.deg == 2:
            # maybe this is faster?
            res = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x ** 2)
        else:
            res = self.coef[:, 0] + self.coef[:, 1] * x
            for i in range(2, self.deg):
                res = res + self.coef[:, i] * (x ** i)

        return res
