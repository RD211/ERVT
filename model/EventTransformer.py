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
    

class PredictionHead(nn.Module):
    def __init__(self, opt_dim):
        super(PredictionHead, self).__init__()
        self.latent_embs_compressor = LatentEmbsCompressor(opt_dim)
        self.linear = nn.Linear(opt_dim, 2)
    
    def forward(self, z):
        z = self.latent_embs_compressor(z)
        z = self.linear(z)
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
        self.FF1_2 = nn.Linear(self.D + 4, self.D)
        self.FF2 = nn.Linear(self.D, self.D)

        self.memory = nn.Parameter(torch.normal(0.0, 0.2, (self.M, self.D)).clip(-2,2), requires_grad=True)

        self.transformer_block = TransformerBlock(
            opt_dim = self.D,
            latent_blocks = args.latent_blocks,
            dropout = args.dropout,
            att_dropout = args.att_dropout,
            heads = args.heads,
            cross_heads = args.cross_heads
        )

        self.positional_emb = nn.Parameter(self.sinusoidal_embedding(300, 4), requires_grad=False)


        self.prediction_head = PredictionHead(self.D)

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

        # Permute to N, B, C, H, W
        x = x.permute(1, 0, 2, 3, 4)
        x = x.reshape(N*B, C, H, W)
        x = self.partition(x)
        x = x.view(N, B, -1, self.P * self.P * C)
        x = self.FF1(x)
        # Add the positional encodings
        pos_embs = self.positional_emb.view(-1, self.positional_emb.shape[-1]).expand(N, B, -1, -1)
        x = torch.cat([x, pos_embs], dim=-1)

        # We do FF1_2
        x = self.FF1_2(x)

        x = x.permute(0, 2, 1, 3) # N, T, B, C

        if memory is None:
            memory = self.memory.unsqueeze(1)
            memory = memory.expand(-1, B, -1)

        outputs = []

        for i in range(N):
            x_t = x[i, :, :, :]
            # We do FF2
            x_skip = x_t
            x_t = self.FF2(x_t)
            x_t = x_t + x_skip

            # We apply the transformer block
            x_t = self.transformer_block(x_t, memory)
            memory = memory + x_t

            # We apply the prediction head
            output = self.prediction_head(x_t)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0).permute(1,0,2), memory
    
    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)