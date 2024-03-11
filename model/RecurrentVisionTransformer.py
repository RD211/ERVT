from functools import reduce
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple, List, Type
from argparse import Namespace
from utils.timer import CudaTimer
from model.base import GLU, MLP, LinearHead, LayerScale


class DWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = False,
                 dws_conv_kernel_size: int = 7,
                 cell_update_dropout: float = 0.2):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x: torch.Tensor, h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous[0] is None:
            # generate zero states
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)

        xh = torch.cat((x, h_tm1), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t
    
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


    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x

class FullSelfAttention(nn.Module):
    def __init__(self, channels: int, partition_size: int, dim_head: int = 32):
        super().__init__()

        self.channels = channels
        self.partition_size = partition_size
        self.ls = LayerScale(channels)
        self.mhsa = SelfAttentionCl(dim=channels, dim_head=dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.size()

        attn_output = self.mhsa(x.view(B,H*W,C)).view(B,H,W,C)  # Apply self-attention
        attn_output = self.ls(attn_output)  # Scale the attention output

        return attn_output

class RVTBlock(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.fsa = FullSelfAttention(channels=args.output_channels, partition_size=args.partition_size, dim_head=args.dim_head)

        self.ln1= nn.LayerNorm(args.output_channels)
        self.ln2= nn.LayerNorm(args.output_channels)

        self.mlpb = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)

        self.lstm = DWSConvLSTM2d(dim=args.output_channels)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln1(x_conv)

        x_bsa = self.fsa(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln2(x_bsa)
        x_bsa = self.mlpb(x_bsa)

        x_unflattened = x_bsa.permute(0, 3, 1, 2)
        x_unflattened, c = self.lstm(x_unflattened, (h, c))

        return x_unflattened, c

class RVT(nn.Module):
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
            RVTBlock(
                replace_act_layer({**(args.__dict__), **args.stages[i]}),
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.in_channels,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)

    def forward(self, x, lstm_states=None):
        B, N, C, H, W = x.size()

        if lstm_states == None:
            lstm_states = [None] * len(self.stages)
        outputs = []

        # We iterate over the time dimension
        for t in range(N):

            # We get the input for the current time step
            xt = x[:, t, :, :, :]

            # For each stage we apply the RVTBlock
            for i, stage in enumerate(self.stages):
                lstm_state = lstm_states[i] if lstm_states[i] is not None else (None, None)
                xt, c = stage(xt, lstm_state[0], lstm_state[1])

                # We save it for the next time step
                lstm_states[i] = (xt, c)


            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1)

        return coordinates, lstm_states
