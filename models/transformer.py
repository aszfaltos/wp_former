import torch
from torch import nn, Tensor
import math
from dataclasses import dataclass


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, r, v, mask=None):
        # outer product of Q and K normalized
        attn_scores = \
            torch.matmul(q, r.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output

    def split_heads(self, x):
        # batch_size, seq_length, d_model
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size,
                      seq_length,
                      self.num_heads,
                      self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # batch_size, num_heads, seq_length, d_k
        batch_size, _, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.d_model)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))

        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFF, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.fc2(self.tanh(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)] + x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attn_heads = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFF(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn_heads(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFF(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


@dataclass
class TransformerParams:
    src_size: int
    tgt_size: int

    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int

    max_seq_length: int

    dropout: float


class Transformer(nn.Module):
    def __init__(self,
                 params: TransformerParams):
        super(Transformer, self).__init__()
        self.W_encoder = nn.Linear(params.src_size, params.d_model)
        self.W_decoder = nn.Linear(params.tgt_size, params.d_model)

        self.positional_encoding = PositionalEncoding(params.d_model, params.max_seq_length)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(params.d_model, params.num_heads, params.d_ff, params.dropout)
            for _ in range(params.num_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(params.d_model, params.num_heads, params.d_ff, params.dropout)
            for _ in range(params.num_layers)
        ])

        self.fc = nn.Linear(params.d_model, params.tgt_size)
        self.dropout = nn.Dropout(params.dropout)

        for p in self.parameters():
            nn.init.normal_(p, 0, .2)

    @staticmethod
    def generate_mask(src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, tgt_len, tgt_len), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask, tgt_mask = Transformer.generate_mask(src.sum(-1), tgt.sum(-1))
        src_pos_encoded = self.positional_encoding(self.W_encoder(self.dropout(src)))
        tgt_pos_encoded = self.positional_encoding(self.W_decoder(self.dropout(tgt)))
        
        enc_output = src_pos_encoded
        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_pos_encoded
        for dec_layer in self.dec_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(self.dropout(dec_output))
        return output
