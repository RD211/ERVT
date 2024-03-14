from functools import reduce
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple, List, Type
from argparse import Namespace
from utils.timer import CudaTimer
from model.base import GLU, MLP, LayerScale
import torch.nn.functional as F

class SelfAttentionCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.drop = nn.Dropout(0.5)


    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x

class UnifiedBlockGridAttention(nn.Module):
    def __init__(self, channels: int, partition_size: int, dim_head: int = 32, mode: str = 'block'):
        super().__init__()
        assert mode in ['block', 'grid'], "Mode must be either 'block' or 'grid'."
        
        self.channels = channels
        self.partition_size = partition_size
        self.mode = mode
        self.ln = nn.LayerNorm(channels)
        self.mhsa = SelfAttentionCl(dim=channels, dim_head=dim_head)

    def partition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Partitions the input tensor based on the selected mode (block or grid).
        """
        B, H, W, C = x.shape
        if self.mode == 'block':
            if H % self.partition_size != 0 or W % self.partition_size != 0:
                raise ValueError(f"Input size must be divisible by the window size. Got: {H}x{W}, window size: {self.partition_size}")
            shape = (B, H // self.partition_size, self.partition_size, W // self.partition_size, self.partition_size, C)
            permute_order = (0, 1, 3, 2, 4, 5)
        else:  # grid mode
            if H % self.partition_size != 0 or W % self.partition_size != 0:
                raise ValueError(f"Input size must be divisible by the grid size. Got: {H}x{W}, grid size: {self.partition_size}")
            shape = (B, self.partition_size, H // self.partition_size, self.partition_size, W // self.partition_size, C)
            permute_order = (0, 2, 4, 1, 3, 5)
        
        x = x.view(shape)
        windows = x.permute(permute_order).contiguous().view(-1, self.partition_size, self.partition_size, C)
        return windows

    def reverse_partition(self, windows: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
        """
        Reverses the partitioning operation to reconstruct the original image shape.
        """
        H, W = img_size
        C = windows.shape[-1]
        if self.mode == 'block':
            shape = (-1, H // self.partition_size, W // self.partition_size, self.partition_size, self.partition_size, C)
            permute_order = (0, 1, 3, 2, 4, 5)
        else:  # grid mode
            shape = (-1, H // self.partition_size, W // self.partition_size, self.partition_size, self.partition_size, C)
            permute_order = (0, 3, 1, 4, 2, 5)
        
        x = windows.view(shape)
        x = x.permute(permute_order).contiguous().view(-1, H, W, C)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies self-attention based on the selected partitioning mode (block or grid).
        """
        B, H, W, C = x.size()

        x = self.ln(x)  # Apply layer normalization
        x = self.partition(x)  # Partition based on mode

        attn_output = self.mhsa(x)  # Apply self-attention
        attn_output = self.reverse_partition(attn_output, (H, W))  # Reverse partitioning

        return attn_output
    
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

        self.linear = nn.Linear(4800, 2)

    def forward(self, x):
        return self.linear(x)
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.2, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Block(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2, groups=args.input_channels)
        self.conv_combed = nn.Conv2d(args.output_channels*2, args.output_channels, kernel_size=7, stride=1, padding=7//2, groups=args.output_channels)
        self.fsa = UnifiedBlockGridAttention(channels=args.output_channels, partition_size=args.partition_size, dim_head=args.dim_head)

        self.ln1 = nn.LayerNorm(args.output_channels)
        self.ln2 = nn.LayerNorm(args.output_channels)
        self.ln3 = nn.LayerNorm(args.output_channels)

        self.drop = DropPath(drop_prob=0.2)

        self.mlpb = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)
        self.conv_h = nn.Conv2d(args.output_channels, args.output_channels, kernel_size=7, stride=1, padding=7//2, groups=args.output_channels)
        self.relu = nn.ReLU()
        self.conv_g = nn.Conv2d(args.output_channels, args.output_channels, kernel_size=7, stride=1, padding=7//2, groups=args.output_channels)

        self.conv_f = nn.Conv2d(args.output_channels, args.output_channels//4, kernel_size=3, stride=1, padding=3//2, groups=args.output_channels//4)
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # We do conv
        x_conv = self.conv(x)
        x_conv = self.relu(x_conv)
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
        x_concat = self.relu(x_concat)
        x_concat = self.ln2(x_concat)

        x_bsa = self.fsa(x_concat)
        x_bsa = self.drop(x_bsa)
        x_bsa = x_bsa + x_concat
        x_bsa = self.relu(x_bsa)
        x_bsa = self.ln3(x_bsa)
        x_bsa = self.mlpb(x_bsa)

        x_bsa = x_bsa.permute(0, 3, 1, 2)
        x_bsa = self.conv_g(x_bsa)
        x_bsa = self.relu(x_bsa)


        h = self.conv_h(x_bsa)
        h = h.permute(0, 2, 3, 1)
        h = self.relu(h)

        x_bsa = self.conv_f(x_bsa)
        x_bsa = self.relu(x_bsa)
        

        return x_bsa, h

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
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else 32,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)

        self.s0 = nn.Conv2d(args.in_channels, 32, kernel_size=7, stride=2, padding=7//2)

    def forward(self, x, prev_states=None):
        B, N, C, H, W = x.size()

        if prev_states == None:
            prev_states = [None] * len(self.stages)
        outputs = []

        # We iterate over the time dimension
        for t in range(N):

            # We get the input for the current time step
            xt = x[:, t, :, :, :]

            xt = self.s0(xt)

            # For each stage we apply the RVTBlock
            for i, stage in enumerate(self.stages):
                xt, h = stage(xt, prev_states[i])

                # We save it for the next time step
                prev_states[i] = h

            # Pool
            xt = xt.permute(0, 2, 3, 1)
            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1)

        return coordinates, prev_states