###################################################################################################
# Final model. This model is inspired by the Recurrent Vision Transformer (RVT) model.
# Arxiv: https://arxiv.org/abs/2212.05598
###################################################################################################

from functools import reduce
import torch
import torch.nn as nn
from argparse import Namespace
import math
from typing import Optional, Tuple, Type

###################################################################################################
# RVT Model
###################################################################################################

class RVTBlock(nn.Module):
    """
    The RVTBlock is the main building block of the RVT model.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)

        # The entry convolution layer.
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.ln1 = nn.LayerNorm(args.output_channels)

        # The Full Self Attention module
        self.fsa = FullSelfAttention(channels=args.output_channels, dim_head=args.dim_head)
        self.ln2 = nn.LayerNorm(args.output_channels)
        self.mlpb = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)

        # The LSTM module
        self.lstm = DWSConvLSTM2d(dim=args.output_channels, dws_conv_kernel_size=args.kernel_size, cell_update_dropout=0.5)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Entry Convolution
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln1(x_conv)

        # Full Self Attention
        x_bsa = self.fsa(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln2(x_bsa)
        x_bsa = self.mlpb(x_bsa)

        # LSTM
        x_unflattened = x_bsa.permute(0, 3, 1, 2)
        x_unflattened, c = self.lstm(x_unflattened, (h, c))

        return x_unflattened, c

class RVT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        args = Namespace(**vars(args), **kwargs)

        # We initialize all stages using their configs
        self.stages = nn.ModuleList([
            RVTBlock(
                {**(args.__dict__), **args.stages[i]},
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.in_channels,
            )
            for i in range(len(args.stages))
        ])

        # The prediction head
        self.detection = LinearHead(args)

    def forward(self, x, lstm_states=None):
        _, N, _, _, _ = x.size()

        # When this is the first prediction
        if lstm_states == None:
            lstm_states = [(None, None)] * len(self.stages)
        

        outputs = []

        # We iterate over the time dimension
        for t in range(N):

            # We get the input for the current time step
            xt = x[:, t, :, :, :]

            # For each stage we apply the RVTBlock
            for i, stage in enumerate(self.stages):
                h_prev, c_prev = lstm_states[i]
                xt, c = stage(xt, h_prev, c_prev)

                # We save it for the next time step
                lstm_states[i] = (xt, c)


            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1)
        return coordinates, lstm_states

class LinearHead(nn.Module):
    """
    FC Layer detection head.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**vars(args), **kwargs)
        self.args = args

        # We calculate the input dimension of the linear layer
        stride_factor = reduce(lambda x, y: x * y, [s["stride"] for s in args.stages])
        dim = int(args.stages[-1]["output_channels"] * ((args.sensor_width * args.spatial_factor) // stride_factor) * ((args.sensor_height * args.spatial_factor) // stride_factor))

        # Simple linear layer
        self.linear = nn.Linear(dim, 2)

    def forward(self, x):
        return self.linear(x)

###################################################################################################
# LSTM Module
###################################################################################################
class DWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim: int,
                 dws_conv_kernel_size: int = 7,
                 cell_update_dropout: float = 0.2):
        super().__init__()
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim)
        
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x: torch.Tensor, h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # If the previous hidden state is None, we initialize it with zeros
        if h_and_c_previous[0] is None:
            # generate zero states
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)

        h_tm1, c_tm1 = h_and_c_previous

        # We concatenate the input and the previous hidden state
        xh = torch.cat((x, h_tm1), dim=1)
        
        # We convolve the concatenated tensor using a depthwise separable convolution and a 1x1 convolution
        xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        
        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t


###################################################################################################
# Self Attention Module
###################################################################################################
    
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

class FullSelfAttention(nn.Module):
    """ Full self-attention augmented with LayerScale """
    def __init__(self, channels: int, dim_head: int = 32):
        super().__init__()

        self.channels = channels
        self.ls = LayerScale(channels)
        self.mhsa = SelfAttentionCl(dim=channels, dim_head=dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.size()

        attn_output = self.mhsa(x.view(B,H*W,C)).view(B,H,W,C)  # Apply self-attention
        attn_output = self.ls(attn_output)  # Scale the attention output

        return attn_output


###################################################################################################
# Additional Modules
###################################################################################################

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace  # Determines whether to perform the operation in-place.
        # Initializing the learnable scaling parameter (gamma) with the specified initial values.
        # The shape of gamma is determined by 'dim', allowing it to scale each feature independently.
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applying the scaling operation. If 'inplace' is True, 'mul_' is used to modify the input tensor directly.
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class GLU(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 channel_last: bool,
                 bias: bool = True):
        super().__init__()
        # Different activation functions / versions of the gated linear unit:
        # - ReGLU:  Relu
        # - SwiGLU: Swish/SiLU
        # - GeGLU:  GELU
        # - GLU:    Sigmoid
        # seem to be the most promising once.
        # Extensive quantitative eval in table 1: https://arxiv.org/abs/2102.11972
        # Section 2 for explanation and implementation details: https://arxiv.org/abs/2002.05202
        # NOTE: Pytorch has a native GLU implementation: https://pytorch.org/docs/stable/generated/torch.nn.GLU.html?highlight=glu#torch.nn.GLU
        proj_out_dim = dim_out*2
        self.proj = nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last else \
            nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)
        self.channel_dim = -1 if channel_last else 1

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.gelu(gate)

class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))

class MLP(nn.Module):
    def __init__(self,
                 dim: int,
                 channel_last: bool,
                 expansion_ratio: int,
                 gated: bool = True,
                 bias: bool = True,
                 drop_prob: float = 0.):
        super().__init__()
        inner_dim = int(dim * expansion_ratio)
        if gated:
            # To keep the number of parameters (approx) constant regardless of whether glu == True
            # Section 2 for explanation: https://arxiv.org/abs/2002.05202
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias) if channel_last else \
                    nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
                nn.GELU(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)
    
