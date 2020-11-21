import torch
import torch.nn as nn


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()

        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (batch_size, input_length, d_model)

        # x: (batch_size, input_length, d_ff)
        x = self.dropout(torch.relu(self.W1(x)))

        # x: (batch_size, input_length, d_model)
        x = self.W2(x)

        return x
