import torch
import torch.nn as nn

#
# Attention(Q, K, V) = softmax( (Q * K^T)/sqrt(d_k) ) * V
#
# head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V )
#
# Multihead(Q, K, V) = Concat(all head_i) * W^O
#
#
# Q K and V are originally passed in with shape: (N, seq_len, embed_size)
# We want to slice them up for multihead attention, so:
#
#              d_model                  d_k                 d_k
#           *-----------*            *-------*           *-------*
#           |           |            |       |           |       |
#   seq_len |           |    d_model |       |   seq_len |       |
#           |     Q     |            | W_i^Q |           | Q_i'  |
#           |           | DOT        |       | =         |       |
#           |           |            |       |           |       |
#           *-----------*            *-------*           *-------*
#
#              d_model                  d_k                 d_k
#           *-----------*            *-------*           *-------*
#           |           |            |       |           |       |
#   seq_len |           |    d_model |       |   seq_len |       |
#           |     K     |            | W_i^K |           | K_i'  |
#           |           | DOT        |       | =         |       |
#           |           |            |       |           |       |
#           *-----------*            *-------*           *-------*
#
#              d_model                  d_k                 d_k
#           *-----------*            *-------*           *-------*
#           |           |            |       |           |       |
#   seq_len |           |    d_model |       |   seq_len |       |
#           |     V     |            | W_i^V |           | V_i'  |
#           |           | DOT        |       | =         |       |
#           |           |            |       |           |       |
#           *-----------*            *-------*           *-------*
#
#              d_k                                           seq_len
#           *-------*               seq_len               *-----------*
#           |       |            *------------*           |           |
#   seq_len |       |        d_k |            |   seq_len |           |
#           | Q_i^Q |            |   K_i'^T   |           |  SCORES   |
#           |       | DOT        |            | =         |           |
#           |       |            *------------*           |           |
#           *-------*                                     *-----------*
#
#              seq_len                  d_k                 d_k   
#           *-----------*            *-------*           *-------*
#           |           |            |       |           |       |
#   seq_len |           |    seq_len |       |   seq_len |       |
#           |  SCORES   |            | V_i'  |           |head_i |
#           |           | DOT        |       | =         |       |
#           |           |            |       |           |       |
#           *-----------*            *-------*           *-------*
#

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
        num_examples = query.shape[0] # number of training examples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values  = values.reshape(num_examples, value_len, self.num_heads, self.head_dim)
        keys    =   keys.reshape(num_examples, key_len,   self.num_heads, self.head_dim)
        queries =  query.reshape(num_examples, key_len,   self.num_heads, self.head_dim)


        # queries shape: (num_examples, query_len, num_heads, head_dim)
        # keys shape:    (num_examples, key_len,   num_heads, head_dim)
        # scores shape:  (num_examples, num_heads, num_heads, key_len )
        # scores = torch.einsum("sqhd, skhd -> shqk", [queries, keys])

        # naive aproach for edu purposes
        scores = torch.zeros(size=(num_examples, self.num_heads, query_len, key_len))
        for s in range(num_examples): # loop over token embeddings in sequence
            for h in range(self.num_heads): # loop over attention heads
                # naive matmul, store score
                for q_idx in range(query_len):
                    for k_idx in range(key_len):
                        # dot product
                        dot_val = 0.0
                        for d_idx in range(self.head_dim): # d_k
                            dot_val += queries[s, q_idx, h, d_idx] * keys[s, k_idx, h, d_idx]
                        scores[s, h, q_idx, k_idx] = dot_val

        # queries shape: (num_examples, query_len, num_heads, head_dim)
        # keys shape:    (num_examples, key_len,   num_heads, head_dim)
        # scores:        (num_examples, num_heads, query_len, key_len)

        # Causality: the transformer must only work with present or previously ween words.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        # Normalize the scores
        attention = torch.softmax(scores / (self.embed_size ** (1/2)), dim=3)
        # attention shape: (num_examples, num_heads, query_len, key_len)

        out = torch.einsum("shql, slhd -> sqhd", [attention, values]).reshape(
            num_examples, query_len, self.heads * self.head_dim
        )

        # attention shape: (num_examples, num_heads, query_len, key_len)
        # values shape: (num_examples, value_len, num_heads, head_dim)
        # out after matrix multiply: (N, query_len, num_heads, head_dim), then
        # we reshape and flatten the last two dimensions.

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
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        num_heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, num_heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):




