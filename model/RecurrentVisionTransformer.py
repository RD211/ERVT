from functools import reduce
import torch
import torch.nn as nn
from typing import Optional, Tuple

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
    
class TorchMHSAWrapperCl(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, bias: bool = True):
        super().__init__()
        # Ensure the input dimension is divisible by the dimension of each head.
        assert dim % dim_head == 0, "Input dimension must be divisible by the dimension of each head."
        
        num_heads = dim // dim_head  # Calculate the number of attention heads.
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        restore_shape = x.shape  # Store the original shape to restore it after attention.
        B, C = restore_shape[0], restore_shape[-1]  # Extract batch size and channel dimensions.
        
        # Reshape input to (B, Seq Len, C) to fit the expected input shape of nn.MultiheadAttention.
        x = x.view(B, -1, C)
        
        # Apply multi-head self-attention. 
        attn_output, _ = self.mha(query=x, key=x, value=x)
        
        # Reshape the output back to the original input shape.
        attn_output = attn_output.reshape(restore_shape)
        
        return attn_output

class UnifiedBlockGridAttention(nn.Module):
    def __init__(self, channels: int, partition_size: int, dim_head: int = 32, mode: str = 'block'):
        super().__init__()
        assert mode in ['block', 'grid'], "Mode must be either 'block' or 'grid'."
        
        self.channels = channels
        self.partition_size = partition_size
        self.mode = mode
        self.ln = nn.LayerNorm(channels)
        self.ls = LayerScale(channels)
        self.mhsa = TorchMHSAWrapperCl(dim=channels, dim_head=dim_head)

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
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int, partition_size: int, dim_head: int = 32):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.block_sa = UnifiedBlockGridAttention(channels=output_channels, partition_size=partition_size, dim_head=dim_head, mode='block')
        
        self.ln = nn.LayerNorm(output_channels)

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

        self.grid_sa = UnifiedBlockGridAttention(channels=output_channels, partition_size=partition_size, dim_head=dim_head, mode='grid')

        # Currently the LayerScale is not working here
        # self.ls1 = LayerScale(output_channels)
        # self.ls2 = LayerScale(output_channels)

        self.lstm = DWSConvLSTM2d(dim=output_channels)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln(x_conv)


        x_bsa = self.block_sa(x_conv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln(x_bsa)
        x_bsa = self.mlp1(x_bsa)
        # For some reason the LayerScale is not working
        # x_bsa = self.ls1(x_bsa)
        

        x_gsa = self.grid_sa(x_bsa)
        x_gsa = x_gsa + x_bsa
        x_gsa = self.ln(x_gsa)
        x_gsa = self.mlp2(x_gsa)
        # x_gsa = self.ls2(x_gsa)


        x_unflattened = x_gsa.permute(0, 3, 1, 2)
        x_unflattened, c = self.lstm(x_unflattened, (c, h))

        return x_unflattened, c

class RVT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_time_bins = args.n_time_bins

        self.stages = nn.ModuleList([
            RVTBlock(
                input_channels=args.stages[i-1]["channels"] if i > 0 else self.n_time_bins,
                output_channels=args.stages[i]["channels"],
                kernel_size=args.stages[i]["kernel_size"],
                stride=args.stages[i]["stride"],
                partition_size=args.stages[i]["partition_size"],
                dim_head=args.stages[i]["dim_head"]
            )
            for i in range(len(args.stages))
        ])

        width = args.sensor_width
        height = args.sensor_height
        spatial_factor = args.spatial_factor
        stride_factor = reduce(lambda x, y: x * y, [s["stride"] for s in args.stages])
        final_size = args.stages[-1]["channels"] * ((width * spatial_factor) // stride_factor) * ((height * spatial_factor) // stride_factor) 
        self.output_layer = nn.Linear(int(final_size), 2)

    def forward(self, x):
        B, N, C, H, W = x.size()

        x = x.permute(0, 1, 3, 4, 2)

        lstm_states = [None] * len(self.stages)
        outputs = []

        # We iterate over the time dimension
        for t in range(N):
            
            # We get the input for the current time step
            xt = x[:, t, :, :, :] 
            xt = xt.permute(0, 3, 1, 2)

            # For each stage we apply the RVTBlock
            for i, stage in enumerate(self.stages):
                lstm_state = lstm_states[i] if lstm_states[i] is not None else (None, None)
                xt, c = stage(xt, lstm_state[0], lstm_state[1])
                
                # We save it for the next time step
                lstm_states[i] = (c, xt)

                
            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.output_layer(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1) 

        return coordinates