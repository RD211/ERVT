import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from functools import reduce
import math

from model.base import MLP, GLU, LinearHead, ConvBNReLU


class VFAttention(nn.Module):
    """
    """
    def __init__(self, in_chan, mid_chn=256, out_chan=128, norm_layer=nn.BatchNorm2d, *args, **kwargs):
        super(VFAttention, self).__init__()
        self.norm_layer = norm_layer
        self.dim = in_chan
        # mid_chn = int(in_chan/2)
        self.w_qs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_ks = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_vs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.latlayer3 = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.init_weight()

    def forward(self, feat, kv):
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
        # spatio-temporal aggregation
        if kv == None:
            kv = torch.zeros_like(f)
        y = torch.matmul(query, f + kv)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        return p_feat, f

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FTBlock(nn.Module):
    """
    Fast transformer block. In comparison with FRT, this block does not contain an LSTM.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        args = Namespace(**args, **kwargs)
        self.conv = nn.Conv2d(args.input_channels, args.output_channels, kernel_size=args.kernel_size, stride=args.stride, padding=args.kernel_size//2)
        self.fast_att = VFAttention(in_chan=args.output_channels, mid_chn=args.output_channels, out_chan=args.output_channels)
        self.ln = nn.LayerNorm(args.output_channels)
        self.mlp = MLP(dim=args.output_channels, channel_last=True,
            expansion_ratio=args.expansion_ratio, act_layer=args.mlp_act_layer, gated = args.mlp_gated, bias = args.mlp_bias,
            drop_prob=args.drop_prob)

    def forward(self, x, kv):
        x_conv = self.conv(x)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = self.ln(x_conv).permute(0, 3, 1, 2)

        x_bsa, past_kv = self.fast_att(x_conv, kv)
        x_bsa = x_bsa + x_conv
        x_bsa = self.ln(x_bsa.permute(0, 2, 3, 1))
        x_bsa = self.mlp(x_bsa)

        x_unflattened = x_bsa.permute(0, 3, 1, 2)

        return x_unflattened, past_kv


class FastTransformer(nn.Module):
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
            FTBlock(
                replace_act_layer({**(args.__dict__), **args.stages[i]}),
                input_channels=args.stages[i-1]["output_channels"] if i > 0 else args.n_time_bins,
            )
            for i in range(len(args.stages))
        ])

        self.detection = LinearHead(args)

    def forward(self, x):
        B, N, C, H, W = x.size()


        kv_states = [None] * len(self.stages)
        outputs = []

        # We iterate over the time dimension
        for t in range(N):

            # We get the input for the current time step
            xt = x[:, t, :, :, :]

            for i, stage in enumerate(self.stages):
                kv_state = kv_states[i] if kv_states[i] is not None else None
                xt, kv = stage(xt, kv_state)

                # We save it for the next time step
                kv_states[i] = kv


            # Flatten the tensor
            xt = torch.flatten(xt, 1)

            # We take the last stage output and feed it to the output layer
            final_output = self.detection(xt)
            outputs.append(final_output)

        coordinates = torch.stack(outputs, dim=1)

        return coordinates
