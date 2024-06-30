import torch
import torch.nn as nn
import torch.nn.functional as F

#多头自注意力机制
class self_attention(nn.Module):
    def __init__(self, emded_size, num_head):
        super(self_attention, self).__init__()

        self.embed_size = emded_size
        self.num_head = num_head
        self.head_dim = self.embed_size // self.num_head

        self.qkv = nn.Linear(emded_size, emded_size * 3)
        self.out = nn.Linear(emded_size, emded_size)

    def forward(self, x):
        batch_size, len_q, embed_size = x.shape

        #qkv (batch_size, len_q, embed_size * 3)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, len_q, 3, self.num_head, int(self.head_dim))
        qkv = qkv.permute(2, 0, 3, 1, 4)

        #q, k, v (batch_size, num_head, len_q, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #attention (batch_size, num_head, len_q, len_q)
        attention = (q @ k.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention = F.softmax(attention, dim=1)

        #attention_score (batch_size, num_head, len_q, embed_size)
        attention_score = (attention @ v)
        attention_score = attention_score.permute(0, 2, 1, 3)

        #attention_score (batch_size, len_q, embed_size)
        attention_score = attention_score.reshape(batch_size, len_q, embed_size)
        attention_score = self.out(attention_score)

        return attention_score


if __name__ == "__main__":
    batch_size = 2
    len_q = 5
    embed_size = 512
    num_head = 8

    x = torch.randn(batch_size, len_q, embed_size)
    attn = self_attention(embed_size, 8)
    output = attn(x)
    print(f'输入的维度为：{x.shape}\n输出的维度为：{output.shape}')













