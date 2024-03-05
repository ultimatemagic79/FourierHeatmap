"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
from typing import Callable, List, Type
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.helpers import checkpoint_seq

__all__ = [
    "resnet18",
    "resnet20",
    "resnet26",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m: nn.Module) -> None:
    # classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


class LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, x, augmix=False):
        for module in self._modules.values():
            x = module(x, augmix=augmix)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, option: str = "A"
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(  # type: ignore
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(  # type: ignore
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                raise ValueError(f"{option} is not supported.")
        else:
            self.shortcut = nn.Sequential()  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block: Type[nn.Module], num_blocks: List[int], num_classes: int = 10
    ) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # type: ignore
            self.in_planes = planes * block.expansion  # type: ignore

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        rep = out.view(out.size(0), -1)
        logit = self.linear(rep)

        return logit


class BaseBasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(BaseBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.bn1_aug = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)
        self.bn2_aug = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

        # some stats
        self.nn_mass = in_planes + out_planes

    def forward(self, x, augmix=False):
        residual = x

        out = self.conv1(x)
        out = self.bn1_aug(out) if augmix else self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2_aug(out) if augmix else self.bn2(out)

        if self.downsample is not None:
            x = self.downsample[0](x)
            residual = self.downsample[2](x) if augmix else self.downsample[1](x)

        out = out.expand_as(residual) + residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):
    def _init_conv(self, module):
        out_channels, _, kernel_size0, kernel_size1 = module.weight.size()
        n = kernel_size0 * kernel_size1 * out_channels
        module.weight.data.normal_(0, math.sqrt(2.0 / n))

    def _init_bn(self, module):
        module.weight.data.fill_(1)
        module.bias.data.zero_()

    def _init_fc(self, module):
        module.weight.data.normal_(mean=0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()

    def _weight_initialization(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self._init_conv(module)
            elif isinstance(module, nn.BatchNorm2d):
                self._init_bn(module)
            elif isinstance(module, nn.Linear):
                self._init_fc(module)

    def _make_block(self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                self.inplanes,
                                planes * block_fn.expansion,
                                kernel_size=1,
                                stride=stride,
                                bias=False,
                            ),
                        ),
                        (
                            "bn",
                            norm2d(
                                group_norm_num_groups,
                                planes=planes * block_fn.expansion,
                            ),
                        ),
                        (
                            "bn_aug",
                            norm2d(
                                group_norm_num_groups,
                                planes=planes * block_fn.expansion,
                            ),
                        ),
                    ]
                )
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                )
            )
        return CustomSequential(*layers)


class ResNetCifar(ResNetBase):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        split_point: str = "layer3",
        group_norm_num_groups: int = None,
        grad_checkpoint: bool = False,
    ):
        super(ResNetCifar, self).__init__()
        self.num_classes = num_classes
        if split_point not in ["layer2", "layer3", None]:
            raise ValueError(f"invalid split position={split_point}.")
        self.split_point = split_point
        self.grad_checkpoint = grad_checkpoint

        # define model.
        self.depth = depth
        if depth % 6 != 2:
            raise ValueError("depth must be 6n + 2:", depth)
        block_nums = (depth - 2) // 6
        block_fn = BaseBasicBlock
        self.block_nums = block_nums
        self.block_fn_name = "Bottleneck" if depth >= 44 else "BasicBlock"

        # define layers.
        self.inplanes = int(16)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=int(16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16))
        self.bn1_aug = norm2d(group_norm_num_groups, planes=int(16))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * block_fn.expansion),
            out_features=self.num_classes,
            bias=False,
        )

        # weight initialization based on layer type.
        self._weight_initialization()
        self.train()

    def forward_features(self, x, augmix=False):
        """Forward function without classifier. Use gradient checkpointing to save memory."""
        x = self.conv1(x)
        x = self.bn1_aug(x) if augmix else self.bn1(x)
        x = self.relu(x)

        if self.grad_checkpoint:
            x = checkpoint_seq(self.layer1, x, preserve_rng_state=True)
            x = checkpoint_seq(self.layer2, x, preserve_rng_state=True)
            if self.split_point in ["layer3", None]:
                x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
        else:
            x = self.layer1(x, augmix)
            x = self.layer2(x, augmix)
            if self.split_point in ["layer3", None]:
                x = self.layer3(x, augmix)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        return x

    def forward_head(self, x, pre_logits: bool = False, augmix=False):
        """Forward function for classifier. Use gridient checkpointing to save memory."""
        if self.split_point == "layer2":
            if self.grad_checkpoint:
                x = checkpoint_seq(self.layer3, x, preserve_rng_state=True)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
            else:
                x = self.layer3(x, augmix)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

        return x if pre_logits else self.classifier(x)

    def forward(self, x, augmix=False):
        x = self.forward_features(x, augmix=augmix)
        x = self.forward_head(x, augmix=augmix)
        return x


def resnet18(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-20.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNetCifar(num_classes, 18, None, None, False)


def resnet26(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-20.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNetCifar(num_classes, 26, None, None, False)


def resnet20(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-20.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-32.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-44.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-56.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-110.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(pretrained: bool = False, num_classes: int = 10) -> nn.Module:
    """factory class of ResNet-1202.
    Parameters
    ----------
    pretrained : bool
        Rreturn pretrained model. Note: currently just raise error.
    num_classes : int
        Number of output class.
    """
    if pretrained:
        raise NotImplementedError("resnet for cifar10 dose not have pretrained models.")
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)
