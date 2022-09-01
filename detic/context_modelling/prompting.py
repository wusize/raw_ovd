import torch
import torch.nn as nn
import math


class ZeroPositionalEncoding(nn.Module):

    def __init__(self,
                 num_words=4,
                 word_dims=512,):
        super(ZeroPositionalEncoding, self).__init__()
        self.num_words = num_words
        self.word_dims = word_dims

    def forward(self, x):

        return x.new_zeros(x.shape[0], self.num_words, self.word_dims)

class SinePositionalEncoding(nn.Module):

    def __init__(self,
                 num_feats=128,
                 num_words=4,
                 word_dims=512,
                 temperature=1.2,
                 scale=2 * math.pi):
        super(SinePositionalEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.scale = scale
        self.pos_proj = nn.Sequential(
            nn.Linear(num_feats * 4, word_dims),
            nn.LayerNorm(word_dims),
            nn.Linear(word_dims, num_words * word_dims))
        self.num_words = num_words
        self.word_dims = word_dims

    def forward(self, x):
        embed = x * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** ((dim_t // 2) - (self.num_feats // 4))
        pos = embed[:, :, None] * dim_t[None, None]
        pos[..., 0::2] = pos[..., 0::2].sin()
        pos[..., 1::2] = pos[..., 1::2].cos()

        assert pos.shape[-1] == self.num_feats

        pos = pos.view(-1, 4 * self.num_feats)

        return self.pos_proj(pos).view(-1, self.num_words, self.word_dims)


class Prompting(nn.Module):

    def __init__(self,
                 num_words=4,
                 word_dims=512):
        super(Prompting, self).__init__()
        self.proj = nn.Linear(1, num_words * word_dims)
        self.num_words = num_words
        self.word_dims = word_dims

    def forward(self, x):
        num_preds = x.shape[0]
        return self.proj(x.new_ones(num_preds, 1)).view(-1, self.num_words, self.word_dims)
