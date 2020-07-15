import torch
import torch.nn as nn
import math

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, num_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.scale = torch.sqrt(torch.tensor([self.d_k], dtype=torch.float, device=device))

    def forward(self, query, key, value, mask=None):
        # query: (batch_size, input_length1, d_model)
        # key: (batch_size, input_length2, d_model)
        # value: (batch_size, input_length2, d_model)
        # mask: src, src, src, src_mask (batch_size, 1, 1, input_length1)
        # mask: tgt, tgt, tgt, tgt_mask (batch_size, 1, input_length1, input_length1)
        # mask: tgt, encoder_src, encoder_src, src_mask (batch_size, 1, input_length1, input_length1)

        batch_size = query.size(0)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)

        # Q: (batch_size, num_heads, input_length1, d_k)
        # K: (batch_size, num_heads, input_length2, d_k)
        # V: (batch_size, num_heads, input_length2, d_v)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_model).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_model).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_model).transpose(1, 2)

        # energy: (batch_size, num_heads, input_length1, input_length2)
        energy = torch.matmul(Q, K.transpose(2, 3)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask, -1e12)

        # attention: (batch_size, num_heads, input_length1, input_length2)
        attention = torch.softmax(energy, dim=-1)

        # x: (batch_size, num_heads, input_length1, d_k)
        x = torch.matmul(self.dropout(attention), V)

        # x: (batch_size, input_length1, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # x: (batch_size, input_length1, d_model)
        x = self.WO(x)

        return x, attention


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


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super(EncoderLayer, self).__init__()

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

        self.self_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, dropout, device)
        self.feed_forward_layer = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask):
        # src: (batch_size, input_length, d_model)
        # src_mask: (batch_size, input_length)

        src_, self_attention = self.self_attention_layer(src, src, src, src_mask)

        # dropout, residual connection, layer normalization
        # src_encoding: (batch_size, input_length, d_model)
        src = self.self_attention_layer_norm(src + self.dropout(src_))

        src_ = self.feed_forward_layer(src)

        src = self.feed_forward_layer_norm(src + self.dropout(src_))

        return src, self_attention


class PositionalEncoding:

    def __init__(self, d_model, max_len, device):
        # pe: (max_len, d_model)
        self.pe = torch.zeros(max_len, d_model, device=device)

        # position: (max_len, 1)
        position = torch.arange(0.0, max_len, device=device).unsqueeze(1)

        # div_term: (d_model // 2)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        # pe: (1, max_src_len, d_model)
        self.pe = self.pe.unsqueeze(0)

    def __call__(self, x):
        # x: (batch_size, input_length, d_model)

        x = x + self.pe[:, x.size(1)]
        return x


class Encoder(nn.Module):

    def __init__(self, src_vocab_size, max_src_len, d_model, num_layers, num_heads, d_ff, dropout, device):
        super(Encoder, self).__init__()

        self.device = device

        self.pos_embedding = PositionalEncoding(d_model, max_src_len, device)
        self.token_embedding = nn.Embedding(src_vocab_size, d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, device)
                                     for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

        self.scale = torch.sqrt(torch.tensor([d_model], dtype=torch.float, device=device))

    def forward(self, src, src_mask):
        # src: (batch_size, input_length)
        # src_mask: (batch_size, input_length)

        # src: (batch_size, input_length, d_model)
        src = self.token_embedding(src) * self.scale

        src = self.dropout(self.pos_embedding(src))

        for layer in self.layers:
            src, self_attention = layer(src, src_mask)

        return src


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super(DecoderLayer, self).__init__()

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

        self.self_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, dropout, device)
        self.encoder_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, dropout, device)
        self.feed_forward_layer = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tgt, encoder_src, tgt_mask, src_mask):
        # tgt: (batch_size, tgt_input_length, d_model)
        # encoder_src: (batch_size, src_input_length, d_model)
        # tgt_mask: (batch_size, 1, tgt_input_length, tgt_input_length)
        # src_mask: (batch_size, 1, 1, src_input_length)

        tgt_, self_attention = self.self_attention_layer(tgt, tgt, tgt, tgt_mask)
        tgt = self.self_attention_layer_norm(tgt + self.dropout(tgt_))

        tgt_, encoder_attention = self.encoder_attention_layer(tgt, encoder_src, encoder_src, src_mask)
        tgt = self.encoder_attention_layer_norm(tgt + self.dropout(tgt_))

        tgt_ = self.feed_forward_layer(tgt)
        tgt = self.feed_forward_layer_norm(tgt + self.dropout(tgt_))

        return tgt, self_attention, encoder_attention


class Decoder(nn.Module):

    def __init__(self, tgt_vocab_size, max_tgt_len, d_model, num_layers, num_heads, d_ff, dropout, device):
        super(Decoder, self).__init__()

        self.device = device

        self.pos_embedding = PositionalEncoding(d_model, max_tgt_len, device)
        self.token_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, device)
                                     for _ in range(num_layers)])

        self.linear = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(p=dropout)

        self.scale = torch.sqrt(torch.tensor([d_model], dtype=torch.float, device=device))

    def forward(self, tgt, encoder_src, tgt_mask, src_mask):
        # tgt: (batch_size, tgt_input_length, d_model)
        # encoder_src: (batch_size, src_input_length, d_model)
        # tgt_mask: (batch_size, tgt_input_length)
        # src_mask: (batch_size, src_input_length)

        tgt = self.token_embedding(tgt) * self.scale

        tgt = self.dropout(self.pos_embedding(tgt))

        for layer in self.layers:
            tgt, self_attention, encoder_attention = layer(tgt, encoder_src, tgt_mask, src_mask)

        # output: (batch_size, tgt_input_length, tgt_vocab_size)
        output = self.linear(tgt)

        return output


class S2S(nn.Module):

    def __init__(self, encoder, decoder, padding_value, device):
        super(S2S, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.padding_value = padding_value

        self.device = device

    def make_src_mask(self, src):
        # src: (batch_size, src_length)

        # src_mask: (batch_size, 1, 1, src_length)
        src_mask = torch.ne(src, self.padding_value).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt: (batch_size, tgt_length)

        # tgt_pad_mask: (batch_size, 1, 1, tgt_length)
        tgt_pad_mask = torch.ne(tgt, self.padding_value).unsqueeze(1).unsqueeze(2)

        tgt_length = tgt.size(1)

        # tgt_sub_mask: (tgt_length, tgt_length)
        tgt_sub_mask = torch.tril(torch.ones(tgt_length, tgt_length, device=self.device)).bool()

        # tgt_mask: (batch_size, 1, tgt_length, tgt_length)
        tgt_mask = torch.logical_and(tgt_pad_mask, tgt_sub_mask)

        return tgt_mask

    def forward(self, src, tgt):
        # src: (batch_size, src_length)
        # tgt: (batch_size, tgt_length)

        # src_mask: (batch_size, 1, 1, src_length)
        # tgt_mask: (batch_size, 1, tgt_length, tgt_length)
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        encoder_src = self.encoder(src, src_mask)

        output, attention = self.decoder(tgt, encoder_src, tgt_mask, src_mask)

        return output, attention

    def init_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def count_parameters(self):

        return sum(p.numel() for p in self.parameters())

    def train_batch(self, input_batch, target_batch, criterion, optimizer):

        # input_batch: (batch_size, src_input_length)
        # target_batch: (batch_size, tgt_input_length)

        # output: (batch_size, tgt_input_length, tgt_vocab_size)
        output, attention = self(input_batch, target_batch)

        batch_loss = criterion(output.transpose(1, 2), target_batch)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        return batch_loss.item()
