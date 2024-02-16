import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

CHANNELS = {
    "T": [32, 64, 128, 256],
    "S": [48, 96, 192, 384],
    "B": [64, 128, 256, 512]
}

KERNELS = [7, 3, 3, 3]
STRIDES = [2, 2, 2, 2]
GRID_SIZE = [2, 5, 2, 2]
NR_OF_STAGES = 2

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

    def forward(self, x: th.Tensor, h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None) \
            -> Tuple[th.Tensor, th.Tensor]:
        """
        :param x: (N C H W)
        :param h_and_c_previous: ((N C H W), (N C H W))
        :return: ((N C H W), (N C H W))
        """
        if h_and_c_previous[0] is None:
            # generate zero states
            hidden = th.zeros_like(x)
            cell = th.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)

        xh = th.cat((x, h_tm1), dim=1)
        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)
        mix = self.conv1x1(xh)

        gates, cell_input = th.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = th.sigmoid(gates)
        forget_gate, input_gate, output_gate = th.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(th.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * th.tanh(c_t)

        return h_t, c_t
    
class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * th.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma
    
class TorchMHSAWrapperCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        assert dim % dim_head == 0
        num_heads = dim // dim_head
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True)

    def forward(self, x: th.Tensor):
        restore_shape = x.shape
        B, C = restore_shape[0], restore_shape[-1]
        x = x.view(B, -1, C)
        attn_output, attn_output_weights =  self.mha(query=x, key=x, value=x)
        attn_output = attn_output.reshape(restore_shape)
        return attn_output

class BlockSelfAttention(nn.Module):
    def __init__(self, channels, window_size):
        super(BlockSelfAttention, self).__init__()
        self.channels = channels
        self.window_size = window_size # P
                
        # layer norm
        self.ln = nn.LayerNorm(channels)

        # layer scale
        self.ls = LayerScale(channels)
        
        self.mhsa = TorchMHSAWrapperCl(dim=channels)

    def window_partition(self, x):
        B, H, W, C = x.shape
        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError("The height and width of the input must be divisible by the window size." + str(self.window_size) + " " + str(H) +"x" +str(W))
        
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows
    
    def window_reverse(self, windows, img_size):
        H, W = img_size
        C = windows.shape[-1]
        x = windows.view(-1, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        return x
    
    def forward(self, x):
        B, H, W, C = x.size()
        P = self.window_size

        # LayerNorm
        x = self.ln(x)

        x = self.window_partition(x)  # B*H*W, P, P, C

        # Apply MHSANow reshape back to the original shape
        attn_output = self.mhsa(x)

        # LayerScale
        attn_output = self.ls(attn_output)

        attn_output = self.window_reverse(attn_output, (H, W))

        return attn_output

class GridAttention(nn.Module):
    def __init__(self, channels, grid_size):
        super(GridAttention, self).__init__()
        self.channels = channels
        self.grid_size = grid_size  # G
        
        # layer norm
        self.ln = nn.LayerNorm(channels)

        # layer scale
        self.ls = LayerScale(channels)

        self.mhsa = TorchMHSAWrapperCl(dim=channels)

    def grid_partition(self, x):
        B, H, W, C = x.shape
        if H % self.grid_size != 0 or W % self.grid_size != 0:
            raise ValueError("The height and width of the input must be divisible by the grid size.")
        
        x = x.view(B, self.grid_size, H // self.grid_size, self.grid_size, W // self.grid_size, C)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, self.grid_size, self.grid_size, C)
        return windows


    def grid_reverse(self, windows, img_size):
        H, W = img_size
        C = windows.shape[-1]
        x = windows.view(-1, H // self.grid_size, W // self.grid_size, self.grid_size, self.grid_size, C)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
        return x
    
    def forward(self, x):
        B, H, W, C = x.size()

        # LayerNorm
        x = self.ln(x)

        # Partition
        x = self.grid_partition(x)

        # Apply MHSANow reshape back to the original shape
        attn_output = self.mhsa(x)

        # LayerScale
        attn_output = self.ls(attn_output)

        attn_output = self.grid_reverse(attn_output, (H, W))
        
        return attn_output


class RVTBlock(nn.Module):
    def __init__(self, stage, n_time_bins = None, model_type="T"):
        super().__init__()
        self.stage = stage
        self.model_type = model_type
        # If stage is not 0 then we use the previous stage channels
        if stage == 0 and n_time_bins is None:
            raise ValueError("n_time_bins must be provided for stage 0")
        
        input_channels = CHANNELS[model_type][stage-1] if stage > 0 else n_time_bins
        output_channels = CHANNELS[model_type][stage]
        kernel_size = KERNELS[stage]
        stride = STRIDES[stage]

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        
        # Block SA
        self.block_sa = BlockSelfAttention(channels=output_channels, window_size=GRID_SIZE[stage])
        
        # LayerNorm
        self.ln = nn.LayerNorm(output_channels)

        # MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, output_channels)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(),
            nn.Linear(output_channels, output_channels)
        )


        # Grid SA
        self.grid_sa = GridAttention(channels=output_channels, grid_size=GRID_SIZE[stage])


        # Layer scale
        self.ls1 = LayerScale(output_channels)
        self.ls2 = LayerScale(output_channels)

        # LSTM
        self.lstm = DWSConvLSTM2d(dim=output_channels)

    def forward(self, x, c, h):
        # Permute from (B, H, W, C) to (B, C, H, W) for conv2d
        x = x.permute(0, 3, 1, 2)
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln(x_conv)


        x_bsa = self.block_sa(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln(x_bsa)
        x_bsa = self.mlp1(x_bsa)
        # x_bsa = self.ls1(x_bsa)
        

        x_gsa = self.grid_sa(x_bsa)
        x_gsa = x_gsa + x_bsa
        x_gsa = self.ln(x_gsa)
        x_gsa = self.mlp2(x_gsa)
        # x_gsa = self.ls2(x_gsa)


        x_unflattened = x_gsa.permute(0, 3, 1, 2)
        x_unflattened, c = self.lstm(x_unflattened, (c, h))
        x_unflattened = x_unflattened.permute(0, 2, 3, 1)

        return x_unflattened, c

    
class RVT(nn.Module):
    def __init__(self, args, model_type='T'):
        super().__init__()
        self.n_time_bins = args.n_time_bins
        self.model_type = model_type

        # Define the RVT stages
        self.stages = nn.ModuleList([
            RVTBlock(stage=i, n_time_bins=self.n_time_bins if i == 0 else None, model_type=model_type) 
            for i in range(NR_OF_STAGES)
        ])

        # Output layer to get coordinates, assuming the output of the last LSTM has 256 channels for the 'T' model
        self.output_layer = nn.Linear(19200, 2)

    def forward(self, x):
        B, N, C, H, W = x.size()
        # Reshape to B,N,H,W,C
        x = x.permute(0, 1, 3, 4, 2)

        # Initialize LSTM states
        lstm_states = [(None,None)]*len(self.stages)
        outputs = []

        # Pass the input through each of the RVT stages
        for t in range(N):  # iterate over timesteps in the sequence
            xt = x[:, t, :, :, :]  # Get the tensor for the current timestep
            for i, stage in enumerate(self.stages):
                lstm_state = lstm_states[i]
                h, c = stage(xt, *lstm_state)
                lstm_states[i] = (c, h.permute(0, 3, 1, 2))
                xt = h
            # sum per batch
            final_output = self.output_layer(xt.view(B, -1))
            outputs.append(final_output)

        # Convert the list of outputs to a tensor
        coordinates = th.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, 2)

        return coordinates