# Copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from polyact import aespa, PolyAct, PolyActPerChannel
# from gate_tmp import GateAESPA

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

__all__ = ["resnet"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.ratio = 0
        self.state = "vanilla"
        self.conv1 = conv3x3(inplanes, planes, stride)

        # self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = aespa(num_pol=3, planes=planes, norm = 'bnv3', lr_scaler=0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = aespa(num_pol=3, planes=planes, norm = 'bnv3', lr_scaler=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.act1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        self.ratio = 0
        self.state = "vanilla"
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.act1 = aespa(num_pol=3, planes=width, norm = 'bnv3', lr_scaler=0.1)
        self.conv2 = conv3x3(width, width, stride)
        self.act2 = aespa(num_pol=3, planes=width, norm = 'bnv3', lr_scaler=0.1)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = aespa(num_pol=3, planes=planes * self.expansion, norm = 'bnv3', lr_scaler=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += residual
        out = self.act3(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        patchify=False,
    ):
        super(ResNet, self).__init__()
        self.ratio = 0
        self.state = "vanilla"
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.multiplier = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        if self.base_width != 64 and (block is BasicBlock):
            self.multiplier = self.base_width // 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if patchify is True:
            self.conv1 = nn.Conv2d(
                4, self.in_planes, kernel_size=4, stride=4, padding=0, bias=False
            )
            self.maxpool = nn.Sequential()

        self.layer1 = self._make_layer(block, int(64 * self.multiplier), layers[0])
        self.layer2 = self._make_layer(
            block,
            int(128 * self.multiplier),
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            int(256 * self.multiplier),
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            int(512 * self.multiplier),
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * self.multiplier) * block.expansion, num_classes)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def set_ratio(self, ratio):
        self.ratio = ratio

    # def transition(self):
    #     self.state = 'transition'
    #     self.aespa1 = Hermite(num_pol=3, planes=self.bn1.num_features)
    #     self.avgpool_s = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    #     for module in self.children():
    #         if hasattr(module, 'transition'):
    #             module.transition()
    #         for name2, module2 in module.named_children():
    #             if hasattr(module2, 'transition'):
    #                 module2.transition()
    #             if filter_conv(name2):
    #                 for param in module2.parameters():
    #                     param.requires_grad = False
    #             for name3, module3 in module2.named_children():
    #                 if filter_conv(name3):
    #                     for param in module3.parameters():
    #                         param.requires_grad = False
    #         for param in self.fc.parameters():
    #             param.requires_grad = False
    def convert_poly(self):
        self.state = "poly"
        fuse_bn_conv(self.conv1, self.bn1)
        self.act_type1 = Hermite(num_pol=3, planes=self.bn1.num_features)
        self.__delattr__("bn1")
        self.__delattr__("relu")
        for module in self.modules():
            stack = []
            if isinstance(module, BasicBlock):
                if not hasattr(module, "convert_poly"):
                    raise RuntimeError("BasicBlock has no convert_poly method")
                # stack.append(module)
                module.convert_poly()

    def fuse_bn_conv(self):
        self.state = "fused"
        fuse_bn_conv(self.conv1, self.bn1)
        self.__delattr__("bn1")
        self.eval()
        for module in self.modules():
            stack = []
            if isinstance(module, BasicBlock):
                if not hasattr(module, "fuse_bn_conv"):
                    raise RuntimeError("BasicBlock has no fuse_bn_conv method")
                # stack.append(module)
                module.fuse_bn_conv()

    def deploy(self):
        if self.state != "transition":
            raise RuntimeError("deploy model after transition is complete")
        self.state = "deploy"

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        # feat_m.append(self.bn1)
        # feat_m.append(self.relu)
        # feat_m.append(self.maxpool)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        feat_m.append(self.fc)
        return feat_m

    def poly_forward(self, x, is_feat=False):
        x = self.conv1(x)
        x = self.act_type1(x)
        x = self.maxpool(x)
        # if self.state == 'vanilla':
        #     x = self.relu(x)
        #     x = self.maxpool(x)
        # elif self.state == 'transition':
        #     x = (1 - self.ratio) * self.maxpool(self.relu(x)) + self.ratio * self.avgpool_s(self.aespa1(x))
        # elif self.state == 'deploy':
        #     x = self.aespa1(x)
        #     # x = self.maxpool(x)
        #     x = self.avgpool_s(x)

        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f5 = x
        x = self.fc(x)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    def fused_forward(self, x, is_feat=False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # if self.state == 'vanilla':
        #     x = self.relu(x)
        #     x = self.maxpool(x)
        # elif self.state == 'transition':
        #     x = (1 - self.ratio) * self.maxpool(self.relu(x)) + self.ratio * self.avgpool_s(self.aespa1(x))
        # elif self.state == 'deploy':
        #     x = self.aespa1(x)
        #     # x = self.maxpool(x)
        #     x = self.avgpool_s(x)

        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f5 = x
        x = self.fc(x)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    def forward(self, x, is_feat=False):
        if self.state == "fused":
            return self.fused_forward(x, is_feat)
        if self.state == "poly":
            return self.poly_forward(x, is_feat)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.state == "vanilla":
            x = self.relu(x)
            x = self.maxpool(x)
        elif self.state == "transition":
            x = (1 - self.ratio) * self.maxpool(
                self.relu(x)
            ) + self.ratio * self.avgpool_s(self.aespa1(x))
        elif self.state == "deploy":
            x = self.aespa1(x)
            # x = self.maxpool(x)
            x = self.avgpool_s(x)

        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f5 = x
        x = self.fc(x)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def wide_resnet10_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet10_2", BasicBlock, [1, 1, 1, 1], pretrained, progress, **kwargs
    )


def wide_resnet18_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet18_2", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def wide_resnet26_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet26_2", BasicBlock, [3, 3, 3, 3], pretrained, progress, **kwargs
    )


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def wide_resnet34_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet34_2", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet34_4(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 4
    return _resnet(
        "wide_resnet34_4", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


if __name__ == "__main__":
    x = torch.randn(64, 3, 224, 224)

    net = wide_resnet10_2()

    feats, logit = net(x, is_feat=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    num_params_stu = sum(p.numel() for p in net.parameters()) / 1000000.0
    print("Total params_stu: {:.3f} M".format(num_params_stu))
