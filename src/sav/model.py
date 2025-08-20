import torch.nn as nn
import torch.nn.init as init
import torch
import math
from utils import get_config, set_seed

config = get_config()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SelfAttention(nn.Module):
    def __init__(self, h, d_model):
        super(SelfAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # perform integer division
        self.d_k_sqrt = self.d_k ** 0.5
        self.h = h
        self.linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(4)])  # 4 Linear

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """

        batch_size = x.size(0)

        # pass x through a layer of Linear transformation to obtain QKV, keeping the tensor size unchanged
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linear[0:3], [x] * 3)]  # [batch_size, h, seq_len, d_k]

        # apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k_sqrt  # [batch_size, h, seq_len, seq_len]
        scores = scores.softmax(dim=-1)

        x = torch.matmul(scores, value)

        # 'concat' using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # [batch_size, seq_len, d_model]
        x = self.linear[-1](x)

        return x, scores


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.fw = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        """
        :param x: torch.Size([batch_size, seq_len, d_model])
        :return result: torch.Size([batch_size, seq_len, d_model])
        """
        return self.fw(x)


class MNISTModel(nn.Module):
    """Predicts AVT, JSC according to the mater_encoding."""

    def __init__(self, patch_size, d_model, h, dropout, N, n_classes):
        super(MNISTModel, self).__init__()
        self.N = N
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_size))  # classification token
        self.linear_map = nn.Linear(patch_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

        self.attentions = nn.ModuleList([SelfAttention(h, d_model) for _ in range(N)])
        self.ln_attn = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])

        self.feedforwards = nn.ModuleList([FeedForward(d_model) for _ in range(N)])
        self.ln_ffd = nn.ModuleList([LayerNorm(d_model) for _ in range(N)])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """
        :param x: torch.size([batch_size, seq_len, patch_size])
        :return classes: torch.size([batch_size, n_classes])
        """
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size, 1, patch_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len+1, patch_size)
        x = self.pe(self.linear_map(x))  # (batch_size, seq_len+1, d_model)

        attention_scores = []  # [N, batch_size, h, seq_len, seq_len]
        for i in range(self.N):
            x, scores = self.attentions[i](x)
            attention_scores.append(scores)
            x = self.ln_attn[i](x + self.dropout(x))  # multi-head attention and layer norm
            x = self.ln_ffd[i](x + self.dropout(self.feedforwards[i](x)))  # feedforward and layer norm

        logit = self.classifier(x[:, 0, :])
        return logit, attention_scores


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0.01)


def get_model():
    model = MNISTModel(
        patch_size=config["data"]["channels"] * config["data"]["patch_size"][0] * config["data"]["patch_size"][1],
        d_model=config["model"]["d_model"],
        h=config["model"]["h"],
        dropout=config["model"]["dropout"],
        N=config["model"]["N"],
        n_classes=config["data"]["n_classes"]
    )
    model.apply(init_weights)
    return model


if __name__ == '__main__':
    set_seed()
    print(f"model parameters: {sum(p.numel() for p in get_model().parameters() if p.requires_grad) / 1000000:.3f}M.")
