import math
import torch
import torch.nn as nn
from argparse import Namespace
import torch.nn.functional as F

class AttentionBlock(nn.Module):    # PerceiverAttentionBlock
    def __init__(self, opt_dim, heads, dropout, att_dropout):
        super(AttentionBlock, self).__init__()

        self.layer_norm_x = nn.LayerNorm([opt_dim])
        self.layer_norm_1 = nn.LayerNorm([opt_dim])
        self.layer_norm_att = nn.LayerNorm([opt_dim])
        
        self.attention = nn.MultiheadAttention(
            opt_dim,            # embed_dim
            heads,              # num_heads
            dropout=att_dropout,
            bias=True,
            add_bias_kv=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.layer_norm_2 = nn.LayerNorm([opt_dim])
        self.linear2 = nn.Linear(opt_dim, opt_dim)
        self.linear3 = nn.Linear(opt_dim, opt_dim)


    def forward(self, x, z_input, mask=None, q_mask=None):
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)
        
        z_att, _ = self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)  # Q, K, V
        
        z_att = z_att + z_input
        z = self.layer_norm_att(z_att)

        z = self.dropout(z)
        z = self.linear1(z)
        z = torch.nn.GELU()(z)

        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = torch.nn.GELU()(z)
        z = self.dropout(z)
        z = self.linear3(z)
        
        return z + z_att

class TransformerBlock(nn.Module):
    def __init__(self, opt_dim, latent_blocks, dropout, att_dropout, heads, cross_heads):
        super(TransformerBlock, self).__init__()

        self.cross_attention = AttentionBlock(opt_dim=opt_dim, heads=cross_heads, dropout=dropout, att_dropout=att_dropout)
        self.latent_attentions = nn.ModuleList([
            AttentionBlock(opt_dim=opt_dim, heads=heads, dropout=dropout, att_dropout=att_dropout) for _ in range(latent_blocks)
        ])

    def forward(self, x_input, z, mask=None, q_mask=None):
        z = self.cross_attention(x_input, z, mask=mask, q_mask=q_mask)
        for latent_attention in self.latent_attentions:
            z = latent_attention(z, z, q_mask=q_mask)
        return z
    
    
class LatentEmbsCompressor(nn.Module):
    
    # Linear + compressor
    def __init__(self, opt_dim):
        super(LatentEmbsCompressor, self).__init__()
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.layer_norm = nn.LayerNorm([opt_dim])
    
    # batch_size x num_latent x emb_dim
    def forward(self, z):
        z = self.layer_norm(z)
        z = self.linear1(z)
        z = F.relu(z)
        z = z.mean(dim=0)
        return z
    

############################################################################################################
# Event Transformer
############################################################################################################
class EVT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        args = Namespace(**vars(args), **kwargs)
        self.args = args
        self.M = args.M
        self.D = args.D
        self.P = args.P
        self.n_time_bins = args.n_time_bins

        self.FF1 = nn.Linear(self.P * self.P * args.n_time_bins, self.D)
        self.FF2 = nn.Linear(self.D, self.D)

        self.transformer_block = TransformerBlock(
            opt_dim = self.D,
            latent_blocks = args.latent_blocks,
            dropout = args.dropout,
            att_dropout = args.att_dropout,
            heads = args.heads,
            cross_heads = args.cross_heads
        )

    def partition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Partitions the input tensor based on the selected mode (block or grid).
        """
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        if H % self.P != 0 or W % self.P != 0:
            raise ValueError(f"Input size must be divisible by the window size. Got: {H}x{W}, window size: {self.P}")
        shape = (B, H // self.P, self.P, W // self.P, self.P, C)
        permute_order = (0, 1, 3, 2, 4, 5)
        x = x.view(shape)
        windows = x.permute(permute_order).contiguous().view(B, -1, self.P, self.P, C)
        return windows
    
    def forward(self, x, memory = None):
        B, N, C, H, W = x.size()

        if memory is None:
            # mean 0.0 and deviation 0.2    
            memory = torch.normal(0.0, 0.2, (B, self.M, self.D)).to(x.device)

        for i in range(N):
            x_t = x[:, i, :, :, :]
            # We split into patches of PxP
            x_t = self.partition(x_t)
            print("We have split the input into patches of PxP this many times: ", x_t.shape)

            # We flatten each patch
            x_t = x_t.flatten(2)
            print("We have flattened each patch this many times: ", x_t.shape)

            # We apply the FF1
            x_t = self.FF1(x_t)
            
            # We add positional encodings
            # x_t = x_t + fourier_features(x_t.shape[1:], self.M)

            # We do FF2
            x_skip = x_t
            x_t = self.FF2(x_t)
            x_t = x_t + x_skip

            print("We have applied the FF1 and FF2 this many times: ", x_t.shape)
            print("We have applied the FF1 and FF2 this many times: ", memory.shape)

            # We apply the transformer block
            memory = self.transformer_block(x_t, memory)

            
            


            


            


def fourier_features(shape, bands):
    # This first "shape" refers to the shape of the input data, not the output of this function
    dims = len(shape)

    # Every tensor we make has shape: (bands, dimension, x, y, etc...)

    # Pos is computed for the second tensor dimension
    # (aptly named "dimension"), with respect to all
    # following tensor-dimensions ("x", "y", "z", etc.)
    pos = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
    )))
    pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

    # Band frequencies are computed for the first
    # tensor-dimension (aptly named "bands") with
    # respect to the index in that dimension
    band_frequencies = (torch.logspace(
        math.log(1.0),
        math.log(shape[0]/2),
        steps=bands,
        base=math.e
    )).view((bands,) + tuple(1 for _ in pos.shape[1:])).expand(pos.shape)

    # For every single value in the tensor, let's compute:
    #             freq[band] * pi * pos[d]

    # We can easily do that because every tensor is the
    # same shape, and repeated in the dimensions where
    # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
    result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

    # Use both sin & cos for each band, and then add raw position as well
    # TODO: raw position
    result = torch.cat([
        torch.sin(result),
        torch.cos(result),
    ], dim=0)

    return result