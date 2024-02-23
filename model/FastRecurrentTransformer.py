from functools import reduce
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple, List, Type
from argparse import Namespace
from utils.timer import CudaTimer
import torch.nn.functional as F

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

class FRTBlock(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.fast_att = FastAttention(in_chan=args.output_channels, mid_chn=args.output_channels, out_chan=args.output_channels)
        self.ln = nn.LayerNorm(args.output_channels)
        self.mlp = MLP(dim=args.output_channels, channel_last=True, expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias, drop_prob=args.drop_prob)
        self.lstm = DWSConvLSTM2d(dim=args.output_channels)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln(x_conv).permute(0, 3, 1, 2)

        x_bsa = self.fast_att(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln(x_bsa.permute(0, 2, 3, 1))
        x_bsa = self.mlp(x_bsa)

        x_unflattened = x_bsa.permute(0, 3, 1, 2)
        x_unflattened, c = self.lstm(x_unflattened, (c, h))

        return x_unflattened, c

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, activation='leaky_relu', *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan)
        else:
            self.bn =  lambda x:x

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FastAttention(nn.Module):
    """
    """
    def __init__(self, in_chan, mid_chn=256, out_chan=128, norm_layer=nn.BatchNorm2d, *args, **kwargs):
        super(FastAttention, self).__init__()
        self.norm_layer = norm_layer
        self.dim = in_chan
        # mid_chn = int(in_chan/2)
        self.w_qs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_ks = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_vs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.latlayer3 = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        #self.init_weight()

    def forward(self, feat):
        # Expect shape N x C x H x W
        query = self.w_qs(feat)
        key   = self.w_ks(feat)
        value = self.w_vs(feat)

        N,C,H,W = feat.size()

        query_ = query.view(N,self.dim,-1).permute(0, 2, 1)
        query = F.normalize(query_, p=2, dim=2, eps=1e-12)

        key_   = key.view(N,self.dim,-1)
        key   = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N,C,-1).permute(0, 2, 1)

        f = torch.matmul(key, value)
        y = torch.matmul(query, f)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        return p_feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

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

class FRT(nn.Module):
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
            FRTBlock(
                replace_act_layer({**(args.__dict__), **args.stages[i]}),
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.n_time_bins,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)

    def forward(self, x):
        B, N, C, H, W = x.size()


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
                lstm_states[i] = (c, xt)


            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1) 

        return coordinates