import math
import torch


class PositionalEncodingLayer:

    def __init__(self, d_model, max_len, device):
        # pe: (max_len, d_model)
        self.pe = torch.zeros(max_len, d_model, device=device)

        # position: (max_len, 1)
        position = torch.arange(0.0, max_len, device=device).unsqueeze(1)

        # div_term: (d_model // 2)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        # pe: (1, max_len, d_model)
        self.pe = self.pe.unsqueeze(0)

    def __call__(self, x):
        # x: (batch_size, input_length, d_model)

        x = x + self.pe[:, :x.size(1)]
        return x
