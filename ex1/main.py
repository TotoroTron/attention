import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads # integer division

        assert(self.head_dim * num_heads == embed_size), "embed_size must be divisible by num_heads"

        #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        #
        self.values     = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys       = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries    = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out     = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding vector into self.heads subvectors
        # N = batch size
        # value_len = length dimension for values
        #
        values  = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys    =   keys.reshape(N, key_len,   self.num_heads, self.head_dim)
        queries =  query.reshape(N, key_len,   self.num_heads, self.head_dim)


        # queries shape:    (N, query_len, num_heads, heads_dim)
        # keys shape:       (N, key_len,   num_heads, heads_dim)
        # energy shape:     (N, num_heads, num_heads, key_len  )
        score = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        # score shape:      (N, N)

        # masked attentions
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-1e20"))

        # Attention
        attention = torch.softmax(score / (self.embed_size ** (1/2)), dim=3)

        # attention shape:  (N, num_heads, query_len, key_len   )
        # values shape:     (N, value_len, num_heads, heads_dim )
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # out shape:        (N, query_len, num_heads, head_dim  ) then flatten last two dimensions

        # Fully connected layer
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        ...

    ...



