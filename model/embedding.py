import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, length: int = 512, d_model: int = 64) -> Tensor:
        super(SinusoidalPositionEmbedding, self).__init__()
        pe = torch.zeros(length, d_model)
        
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input: Tensor) -> Tensor:
        return self.pe[:, :input.size(1)].to(input.device)


class ParamixerEmbedding(nn.Module):
    def __init__(self, pe_type, pooling_type, vocab_size, max_seq_len, embed_dim, 
                 embed_drop_prob) -> None:
        super(ParamixerEmbedding, self).__init__()
        assert pe_type in ['nope', 'spe', 'ape']

        self.pe_type = pe_type
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        if pooling_type == "CLS":
            padding_idx = vocab_size - 2
        else:
            padding_idx = None
        self.token_embed = nn.Embedding(
                            vocab_size,
                            embed_dim,
                            padding_idx
                        )
        self.pos_embed = nn.Embedding(
                            max_seq_len, 
                            embed_dim
                        )
        self.sin_pos_embed = SinusoidalPositionEmbedding(max_seq_len, embed_dim)
        self.embed_dropout = nn.Dropout(p=embed_drop_prob)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.token_embed.weight, a=-0.1, b=0.1)
        init.uniform_(self.pos_embed.weight, a=-0.1, b=0.1)
    
    def forward(self, input: Tensor) -> Tensor:
        token_embed = self.token_embed(input)
        if self.pe_type == 'nope':
            # No Position Embedding
            embed = token_embed
        elif self.pe_type == 'spe':
            # Sinusoidal Positional Encoding
            pos_embed = self.sin_pos_embed(token_embed)
            embed = token_embed + pos_embed
        elif self.pe_type == 'ape':
            # Absolute Learnable Position Embedding
            pos_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input.device)
            pos_ids = pos_ids.expand(input.size(0), self.max_seq_len)
            pos_embed = self.pos_embed(pos_ids)
            embed = token_embed + pos_embed
        else:
            raise ValueError(f'ERROR: The Position Embedding {self.pe} is not implemented yet.')
        
        embed = self.embed_dropout(embed)

        return embed