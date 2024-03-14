import torch
import torch.nn as nn
import torch.nn.functional as F
from argspace import Namespace

from .base import LinearHead


class GroupTBlock(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()

        args = Namespace(**args, **kwargs)

        # group convolution
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride,
                padding = args.kernel_size // 2, groups = args.cgroups
                )
        self.convln = nn.LayerNorm(args.output_channels)






class GroupTransformer(nn.Module):


    def __init__(self, args, **kwargs):

        super().__init__()

        args = Namespace(**vars(args), **kwargs):
        self.args = args

        act_layer_to_activation = {
                "gelu": nn.GELU,
                "relu": nn.ReLU
        }

        def replace_act_layer(args):
            args["mlp_act_layer"] = act_layer_to_activation[args["mlp_act_layer"]]
            return args 

        self.stages = nn.ModuleList([
            GroupTBlock(
                replace_act_layer({**(args.__dict__), **args.stages[i]}),
                input_channels = args.stages[i - 1]["output_channels"] if i > 0 else args.in_channels
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)


    def forward(self, x, lstm_states=None):
        B, N, C, H, W = x.size()

        if lstm_states == None:
            lstm_states = [None] * len(self.args)

        outputs = []

        for t in range(N):

            xt = x[:, t, :, :, :]

            for i, stage in enumerate(self.stages):
                lstm_state = lstm_states[i] if lstm_states[i] is not None else (None, None)
                xt, c = stage(xt, lstm_state[0], lstm_state[1])

                lstm_states[i] = (xt, c)

            xt = torch.flatten(xt, 1)

            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, 1)

        return coordinates, lstm_states

