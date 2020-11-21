import torch
import torch.nn as nn

from models.MultiHeadAttention import MultiHeadAttentionLayer
from models.PositionalEncoding import PositionalEncodingLayer
from models.PositionwiseFeedforward import PositionwiseFeedforwardLayer
from models.MultiHeadAttention import MultiHeadAttentionLayerRPE


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, encoder_max_rpe, max_src_len, dropout, device):
        super(EncoderLayer, self).__init__()

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

        self.self_attention_layer = MultiHeadAttentionLayerRPE(d_model, num_heads, encoder_max_rpe, max_src_len,
                                                               dropout, device)
        self.feed_forward_layer = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask, return_attention=False):

        # parameter return_attention can only be used during evaluation

        # src: (batch_size, input_length, d_model)
        # src_mask: (batch_size, 1, 1, src_length)

        src_ = self.self_attention_layer(src, src, src, src_mask)
        del src_mask

        # dropout, residual connection, layer normalization
        # src: (batch_size, input_length, d_model)
        src = self.self_attention_layer_norm(src + self.dropout(src_[0]))

        if self.training:
            del src_

        src = self.feed_forward_layer_norm(src + self.dropout(self.feed_forward_layer(src)))

        if return_attention:
            return src, src_[1]

        return src


class Encoder(nn.Module):

    def __init__(self, src_vocab_size, max_src_len, d_model, num_layers, num_heads, d_ff, encoder_max_rpe,
                 dropout, device):
        super(Encoder, self).__init__()

        self.device = device

        if encoder_max_rpe > 0:
            self.pos_embedding = None
        else:
            self.pos_embedding = PositionalEncodingLayer(d_model, max_src_len, device)

        self.token_embedding = nn.Embedding(src_vocab_size, d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, encoder_max_rpe, max_src_len,
                                                  dropout, device)
                                     for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

        self.scale = torch.sqrt(torch.tensor([d_model], dtype=torch.float, device=device))

    def forward(self, src, src_mask):
        # src: (batch_size, input_length)
        # src_mask: (batch_size, 1, 1, src_length)

        # src: (batch_size, input_length, d_model)
        src = self.token_embedding(src) * self.scale

        if self.pos_embedding:
            src = self.pos_embedding(src)

        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, decoder_max_rpe, max_tgt_len, dropout, device):
        super(DecoderLayer, self).__init__()

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

        self.self_attention_layer = MultiHeadAttentionLayerRPE(d_model, num_heads, decoder_max_rpe, max_tgt_len,
                                                               dropout, device)
        self.encoder_attention_layer = MultiHeadAttentionLayerRPE(d_model, num_heads, decoder_max_rpe, max_tgt_len,
                                                                  dropout, device)
        self.feed_forward_layer = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tgt, encoder_src, tgt_mask, src_mask, return_attention=False):

        # parameter return_attention can only be used during evaluation

        # tgt: (batch_size, tgt_input_length, d_model)
        # encoder_src: (batch_size, src_input_length, d_model)
        # tgt_mask: (batch_size, 1, tgt_input_length, tgt_input_length)
        # src_mask: (batch_size, 1, 1, src_input_length)

        tgt_ = self.self_attention_layer(tgt, tgt, tgt, tgt_mask)
        del tgt_mask

        if not self.training:
            self_attention = tgt_[1]

        tgt = self.self_attention_layer_norm(tgt + self.dropout(tgt_[0]))
        del tgt_

        tgt_ = self.encoder_attention_layer(tgt, encoder_src, encoder_src, src_mask)
        del encoder_src
        del src_mask

        if not self.training:
            encoder_attention = tgt_[1]

        tgt = self.encoder_attention_layer_norm(tgt + self.dropout(tgt_[0]))
        del tgt_

        tgt = self.feed_forward_layer_norm(tgt + self.dropout(self.feed_forward_layer(tgt)))

        if return_attention:
            return tgt, self_attention, encoder_attention

        return tgt


class Decoder(nn.Module):

    def __init__(self, tgt_vocab_size, max_tgt_len, d_model, num_layers, num_heads, d_ff, share_dec_pro_emb,
                 decoder_max_rpe, dropout, device):
        super(Decoder, self).__init__()

        self.device = device

        if decoder_max_rpe > 0:
            self.pos_embedding = None
        else:
            self.pos_embedding = PositionalEncodingLayer(d_model, max_tgt_len, device)

        self.token_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, decoder_max_rpe, max_tgt_len,
                                                  dropout, device)
                                     for _ in range(num_layers)])

        self.linear = nn.Linear(d_model, tgt_vocab_size)
        if share_dec_pro_emb:
            self.linear.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(p=dropout)

        self.scale = torch.sqrt(torch.tensor([d_model], dtype=torch.float, device=device))

    def forward(self, tgt, encoder_src, tgt_mask, src_mask):
        # tgt: (batch_size, tgt_input_length, d_model)
        # encoder_src: (batch_size, src_input_length, d_model)
        # tgt_mask: (batch_size, 1, tgt_length, tgt_length)
        # src_mask: (batch_size, 1, 1, src_length)

        tgt = self.token_embedding(tgt) * self.scale

        if self.pos_embedding:
            tgt = self.pos_embedding(tgt)

        tgt = self.dropout(tgt)

        for layer in self.layers:
            tgt = layer(tgt, encoder_src, tgt_mask, src_mask)

        # tgt: (batch_size, tgt_input_length, tgt_vocab_size)
        tgt = self.linear(tgt)

        return tgt


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
        del tgt

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

        output = self.decoder(tgt, self.encoder(src, src_mask), tgt_mask, src_mask)

        return output

    def init_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def count_parameters(self):

        return sum(p.numel() for p in self.parameters())

    def train_batch(self, input_batch, target_batch, criterion, optimizer, steps, update_freq):

        if update_freq == 1:
            need_update = True
        else:
            need_update = True if (steps + 1) % update_freq == 0 else False

        """training api used only for single GPU"""

        # input_batch: (batch_size, src_input_length)
        # target_batch: (batch_size, tgt_input_length)

        # output: (batch_size, tgt_input_length - 1, tgt_vocab_size)
        output = self(input_batch, target_batch[:, :-1])
        del input_batch

        # output: (batch_size * (tgt_input_length - 1), tgt_vocab_size)
        output = output.view(-1, output.size(-1))

        # target_batch: (batch_size * (tgt_input_length - 1))
        target_batch = target_batch[:, 1:].contiguous().view(-1)

        batch_loss = criterion(output, target_batch)
        del output
        del target_batch

        batch_loss.backward()

        if need_update:
            optimizer.step()
            optimizer.zero_grad()

        return batch_loss.item()
