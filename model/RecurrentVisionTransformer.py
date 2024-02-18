from functools import reduce
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple, List, Type
from argparse import Namespace
from utils.timer import CudaTimer

class DWSConvLSTM2d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format.
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = False,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.):
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

class GLU(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 channel_last: bool,
                 act_layer: Type[nn.Module],
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

        self.act_layer = act_layer()

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.act_layer(gate)

class MLP(nn.Module):
    def __init__(self,
                 dim: int,
                 channel_last: bool,
                 expansion_ratio: int,
                 act_layer: Type[nn.Module],
                 gated: bool = True,
                 bias: bool = True,
                 drop_prob: float = 0.):
        super().__init__()
        inner_dim = int(dim * expansion_ratio)
        if gated:
            # To keep the number of parameters (approx) constant regardless of whether glu == True
            # Section 2 for explanation: https://arxiv.org/abs/2002.05202
            #inner_dim = round(inner_dim * 2 / 3)
            #inner_dim = math.ceil(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            #inner_dim = round(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, act_layer=act_layer, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias) if channel_last else \
                    nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
                act_layer(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)
    
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

class UnifiedBlockGridAttention(nn.Module):
    def __init__(self, channels: int, partition_size: int, dim_head: int = 32, mode: str = 'block'):
        super().__init__()
        assert mode in ['block', 'grid'], "Mode must be either 'block' or 'grid'."
        
        self.channels = channels
        self.partition_size = partition_size
        self.mode = mode
        self.ln = nn.LayerNorm(channels)
        self.ls = LayerScale(channels)
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
        attn_output = self.ls(attn_output)  # Scale the attention output
        attn_output = self.reverse_partition(attn_output, (H, W))  # Reverse partitioning

        return attn_output

class RVTBlock(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.block_sa = UnifiedBlockGridAttention(channels=args.output_channels, partition_size=args.partition_size, dim_head=args.dim_head, mode='block')
        
        self.ln = nn.LayerNorm(args.output_channels)

        self.mlpb = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)
        self.mlpg = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)

        self.grid_sa = UnifiedBlockGridAttention(channels=args.output_channels, partition_size=args.partition_size, dim_head=args.dim_head, mode='grid')

        self.ls1 = LayerScale(args.output_channels)
        self.ls2 = LayerScale(args.output_channels)

        self.lstm = DWSConvLSTM2d(dim=args.output_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln(x_conv)

        x_bsa = self.block_sa(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln(x_bsa)
        x_bsa = self.mlpb(x_bsa)

        x_gsa = self.grid_sa(x_bsa)
        x_gsa = x_gsa + x_bsa
        x_gsa = self.ln(x_gsa)
        x_gsa = self.mlpg(x_gsa)

        return x_gsa.permute(0, 3, 1, 2)
    
    def forward_with_lstm(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, c = self.lstm(x, (c, h))
        return x, c

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

        self.linear = nn.Linear(dim, 2)

    def forward(self, x):
        return self.linear(x)

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
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.n_time_bins,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)

    def forward(self, x):
        B, N, C, H, W = x.size()

        # Initial LSTM states
        lstm_states = (None, None)

        # Outputs that are to be fed in the detection head
        accumulated_outputs = []

        # We process the tensor column wise, i.e. finishing each stage before moving to the next one for each time step in parallel
        with CudaTimer(device=x.device, timer_name="RVT Time step loop"):

            for i, stage in enumerate(self.stages):

                # If we are at the first stage we use the input tensor, otherwise we use the output from the previous stage
                # We reshape it into B*N, C, H, W to process all time steps in parallel
                xts = x.view(B*N, C, H, W) if i == 0 else xts.view(B*N, xts.size(2), xts.size(3), xts.size(4))

                # Process the stage
                xts = stage(xts)

                # We get the new output shape and reshape it back to B, N, C, H, W
                _, C, H, W = xts.size()
                xts = xts.view(B, N, C, H, W)

                # We now process the time steps sequentially but only the lstm part.
                for t in range(N):
                    xt = xts[:, t, :, :, :]
                    xt, c = stage.forward_with_lstm(xt, lstm_states[0], lstm_states[1])
                    lstm_states = (c, xt)
                    
                    # If we are at the last stage we flatten the tensor and append it to the accumulated outputs
                    if i == len(self.stages) - 1:
                        accumulated_outputs.append(torch.flatten(xt, 1))
                
                # We reset them back.
                lstm_states = (None, None)
                    
        with CudaTimer(device=x.device, timer_name="RVT Detection head"):
            # Stack accumulated outputs along the time dimension and then apply the linear head in parallel
            stacked_outputs = torch.stack(accumulated_outputs, dim=1)
            final_output = self.detection(stacked_outputs.view(B*N, -1))
            coordinates = final_output.view(B, N, -1)

        return coordinates