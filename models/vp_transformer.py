from .vp_layers import vp_layer, ada_hermite
import torch
from torch import nn
from .transformer import Transformer, TransformerParams
from dataclasses import dataclass


@dataclass
class VPTransformerParams:
    src_size: int
    tgt_size: int

    vp_bases: int
    vp_penalty: int

    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int

    max_seq_length: int

    dropout: float


class VPTransformer(nn.Module):
    def __init__(self, params: VPTransformerParams):
        super(VPTransformer, self).__init__()
        self.vp_layer = vp_layer(ada_hermite, params.src_size, params.vp_bases, 2, params.d_model)
        t_params = TransformerParams(
            src_size=params.src_size,
            tgt_size=params.tgt_size,
            d_model=params.d_model,
            num_heads=params.num_heads,
            num_layers=params.num_layers,
            d_ff=params.d_ff,
            max_seq_length=params.max_seq_length,
            dropout=params.dropout
        )
        self.transformer = Transformer(t_params)

    def forward(self, src, tgt, return_vp_and_attn=False):
        # print('src', src.shape)
        vp_out = self.vp_layer.forward(src)
        # print('vp_out', vp_out.shape)
        if not return_vp_and_attn:
            return self.transformer.forward(vp_out, tgt)

        t_out, enc_attn_probs, dec_self_attn_probs, dec_cross_attn_probs = self.transformer.forward(vp_out, tgt,
                                                                                                    return_attn=True)

        return t_out, enc_attn_probs, dec_self_attn_probs, dec_cross_attn_probs, vp_out
