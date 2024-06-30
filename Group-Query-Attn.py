import torch
import torch.nn as nn
import torch.nn.functional as F

class Group_Query_Attention(nn.Module):
    def __init__(self, embed_dim, num_head, num_group):
        super(Group_Query_Attention, self).__init__()
        self.head_dim = embed_dim // num_head
        self.kv_head = num_group
        self.q_head = num_head

        self.wq = nn.Linear(embed_dim, self.q_head * self.head_dim)
        self.wk = nn.Linear(embed_dim, self.kv_head * self.head_dim)
        self.wv = nn.Linear(embed_dim, self.kv_head * self.head_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def expand(self, data):
        repeat_times = self.q_head // self.kv_head
        batch_size, len_q, num_head, head_dim = data.shape
        data_list = []
        for i in range(repeat_times):
            data_list.append(data)
        data = torch.concat(data_list, 2)
        data.view(batch_size, len_q, num_head * repeat_times, head_dim)
        return data

    def forward(self, x):
        b_s, len_q, embed_dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        #xq:(b_s, len_q, q_head * head_dim)
        #xk:(b_s, len_q, k_head * head_dim)
        #xv:(b_s, len_q, v_head * head_dim)

        xq = xq.view(b_s, len_q, self.q_head, self.head_dim)
        xk = xk.view(b_s, len_q, self.kv_head, self.head_dim)
        xv = xv.view(b_s, len_q, self.kv_head, self.head_dim)

        xk = self.expand(xk)
        xv = self.expand(xv)

        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)

        atten_score = xq @ xk.transpose(-2, -1) / (self.head_dim ** 0.5)
        atten_score = F.softmax(atten_score, dim=1)

        #atten_score(b_s, num_head, len_q, len_q)
        attn = atten_score @ xv

        #attn(b_s, num_head, len_q, head_dim)
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, len_q, embed_dim)
        attn = self.out(attn)

        return attn


batch_size = 2
len_n = 128
embed_dim = 512
num_head = 8
num_group = 4

x = torch.randn(batch_size, len_n, embed_dim)

model = Group_Query_Attention(embed_dim, num_head, num_group)
out = model(x)
print(out.shape)








