from functools import reduce
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple, List, Type
from argparse import Namespace
from model.RecurrentVisionTransformer import FullSelfAttention
from utils.timer import CudaTimer
from model.base import GLU, MLP, LayerScale
import torch.nn.functional as F



class LinearHead(nn.Module):
    """
    FC Layer detection head.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**vars(args), **kwargs)
        self.args = args

        stride_factor = reduce(lambda x, y: x * y, [s["stride"] for s in args.stages])
        dim = int(args.stages[-1]["output_channels"] * ((args.sensor_width * args.spatial_factor) // stride_factor) * ((args.sensor_height * args.spatial_factor) // stride_factor))

        self.linear = nn.Linear(19200, 2)

    def forward(self, x):
        return self.linear(x)
    
class Block(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.conv_combed = nn.Conv2d(args.output_channels*2, args.output_channels, kernel_size=7, stride=1, padding=7//2, groups=args.output_channels)
        self.fsa = FullSelfAttention(channels=args.output_channels, partition_size=args.partition_size, dim_head=args.dim_head)

        self.ln1 = nn.LayerNorm(args.output_channels)
        self.ln2 = nn.LayerNorm(args.output_channels)
        self.ln3 = nn.LayerNorm(args.output_channels)

        self.mlpb = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)


    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # We do conv
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln1(x_conv)

        # We concat with h
        if h == None:
            h = torch.zeros_like(x_conv)
        x_concat = torch.cat((x_conv, h), dim=3)

        # We do conv
        x_concat = x_concat.permute(0, 3, 1, 2)
        x_concat = self.conv_combed(x_concat)
        x_concat = x_concat.permute(0, 2, 3, 1)
        x_concat = self.ln2(x_concat)

        x_bsa = self.fsa(x_concat)
        x_bsa = x_bsa + x_concat
        x_bsa = self.ln3(x_bsa)
        x_bsa = self.mlpb(x_bsa)


        return x_bsa.permute(0, 3, 1, 2), x_bsa

class SVT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        args = Namespace(**vars(args), **kwargs)
        self.args = args

        act_layer_to_activation = {
            "gelu": nn.GELU,
            "relu": nn.ReLU
        }

        def replace_act_layer(args):
            args["mlp_act_layer"] = act_layer_to_activation[args["mlp_act_layer"]]
            return args
        self.stages = nn.ModuleList([
            Block(
                replace_act_layer({**(args.__dict__), **args.stages[i]}),
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.in_channels,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)

    def forward(self, x, prev_states=None):
        B, N, C, H, W = x.size()

        if prev_states == None:
            prev_states = [None] * len(self.stages)
        outputs = []

        # We iterate over the time dimension
        for t in range(N):

            # We get the input for the current time step
            xt = x[:, t, :, :, :]

            # For each stage we apply the RVTBlock
            for i, stage in enumerate(self.stages):
                xt, h = stage(xt, prev_states[i])

                # We save it for the next time step
                prev_states[i] = h


            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1)

        return coordinates, prev_states
