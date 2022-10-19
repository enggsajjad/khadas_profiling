# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import math

import torch
import torch.nn as nn
from layers import (BatchNorm2d, Conv2d, FrozenBatchNorm2d, interpolate)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


# include all the operations
PRIMITIVES = {
    "none": lambda C_in, C_out, expansion, stride, **kwargs: Zero(
        stride
    ),
    "skip": lambda C_in, C_out, expansion, stride, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ir_k3_re": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=3, nl="relu", **kwargs
    ),
    "ir_k3_hs": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=3, nl="hswish", **kwargs
    ),
    "ir_k3_r2_re": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=3, nl="relu", dil=2, **kwargs
    ),
    "ir_k3_r2_hs": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=3, nl="hswish", dil=2, **kwargs
    ),
    "ir_k3_r3_re": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=3, nl="relu", dil=3, **kwargs
    ),
    "ir_k5_re": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=5, nl="relu", **kwargs
    ),
    "ir_k5_hs": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=5, nl="hswish", **kwargs
    ),
    "ir_k5_r2_re": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=5, nl="relu", dil=2, **kwargs
    ),
    "ir_k5_r2_hs": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=5, nl="hswish", dil=2, **kwargs
    ),
    "ir_k5_r3_re": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=5, nl="relu", dil=3, **kwargs
    ),
    "ir_k7_re": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=7, nl="relu", **kwargs
    ),
    "ir_k7_hs": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, stride, expansion, kernel=7, nl="hswish", **kwargs
    ),
}


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.output_depth = C_out
        self.conv = (
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(N, C, H, W)
        )


class ConvBNRelu(nn.Sequential):
    def __init__(
            self,
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            no_bias,
            use_relu,
            bn_type,
            group=1,
            dil=1,
            quant=False,
            *args,
            **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", "hswish", None, False]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]
        assert dil in [1, 2, 3, None]

        op = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dil,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )

        # if quant:
        #     op = QuanConv2d(op,
        #                     quan_w_fn=quantizer(CONFIG_SUPERNET['quan']['weight']))

        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            act = nn.ReLU(inplace=True)
            # if quant:
            #     act = QuanAct(act, quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))
            self.add_module("relu", act)
        elif use_relu == "hswish":
            act = nn.Hardswish(inplace=True)
            # if quant:
            #     act = QuanAct(act, quan_a_fn=quantizer(CONFIG_SUPERNET['quan']['act']))
            self.add_module("hswish", act)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )


class IRFBlock(nn.Module):
    def __init__(
            self,
            input_depth,
            output_depth,
            stride,
            expansion=6,
            bn_type="bn",
            kernel=3,
            nl="relu",
            dil=1,
            width_divisor=1,
            shuffle_type=None,
            pw_group=1,
            se=False,
            dw_skip_bn=False,
            dw_skip_relu=False,
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=nl,
            bn_type=bn_type,
            group=pw_group,
        )

        # dw
        self.dw = ConvBNRelu(
            mid_depth,
            mid_depth,
            kernel=kernel,
            stride=stride,
            pad=(kernel // 2) * dil,
            dil=dil,
            group=mid_depth,
            no_bias=1,
            use_relu=nl if not dw_skip_relu else None,
            bn_type=bn_type if not dw_skip_bn else None,
        )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=False,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)

        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x

        return y


if __name__ == '__main__':

    # input = torch.randn(1, 3, 320, 320)
    # for k in PRIMITIVES:
    #     op = PRIMITIVES[k](3, 16, 6, 1)
    #     #print(op)
    #     torch.onnx.export(op,  # model being run
    #                       input,  # model input (or a tuple for multiple inputs)
    #                       "./onnx/{}.onnx".format(k),  # where to save the model (can be a file or file-like object)
    #                       export_params=True,  # store the trained parameter weights inside the model file
    #                       # the ONNX version to export the model to
    #                       opset_version=7,
    #                       verbose=True,
    #                       input_names=['input'],  # the model's input names
    #                       output_names=['output'],  # the model's output names
    #                       dynamic_axes=None,
    #                      )
    
    import netron
    netron.start('./onnx/ir_k3_r2_hs.onnx')