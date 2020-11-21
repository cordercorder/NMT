import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, num_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

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

        del query
        del key
        del value

        # Q: (batch_size, num_heads, input_length1, d_k)
        # K: (batch_size, num_heads, input_length2, d_k)
        # V: (batch_size, num_heads, input_length2, d_v)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # attention: (batch_size, num_heads, input_length1, input_length2)
        attention = torch.matmul(Q, K.transpose(2, 3)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == False, -1e12)

        del mask

        # attention: (batch_size, num_heads, input_length1, input_length2)
        attention = torch.softmax(attention, dim=-1)

        # x: (batch_size, num_heads, input_length1, d_k)
        x = torch.matmul(self.dropout(attention), V)

        # for memory reduction, do not return attention weight during training
        if self.training:
            del attention

        # x: (batch_size, input_length1, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # x: (batch_size, input_length1, d_model)
        x = self.WO(x)

        if self.training:
            return x,
        else:
            return x, attention


class MultiHeadAttentionLayerRPE(nn.Module):

    def __init__(self, d_model, num_heads, max_relative_position, max_cache_len, dropout, device):
        super(MultiHeadAttentionLayerRPE, self).__init__()
        assert d_model % num_heads == 0 and max_relative_position >= 0 and max_cache_len >= 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)

        if max_relative_position > 0:
            self.relative_embedding_k = RelativePosition(self.d_k, max_relative_position, max_cache_len)
        else:
            self.relative_embedding_k = None

        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.tensor([self.d_k], dtype=torch.float, device=device))

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)

        del query
        del key
        del value

        # Q: (batch_size, num_heads, input_length1, d_k)
        # K: (batch_size, num_heads, input_length2, d_k)
        # V: (batch_size, num_heads, input_length2, d_v)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # attention: (batch_size, num_heads, input_length1, input_length2)
        attention = torch.matmul(Q, K.transpose(2, 3))

        if self.relative_embedding_k:
            # attention: (batch_size * num_heads, input_length1, input_length2)
            attention = attention.view(-1, tgt_len, src_len)
            Q = Q.permute(2, 0, 1, 3).contiguous().view(tgt_len, batch_size * self.num_heads, -1)
            attention += Q.bmm(self.relative_embedding_k(tgt_len, src_len).transpose(1, 2)).transpose(0, 1)
            attention = attention.view(batch_size, self.num_heads, tgt_len, src_len)

        attention = attention / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == False, -1e12)

        del mask

        # attention: (batch_size, num_heads, input_length1, input_length2)
        attention = torch.softmax(attention, dim=-1)

        # x: (batch_size, num_heads, input_length1, d_k)
        x = torch.matmul(self.dropout(attention), V)

        # for memory reduction, do not return attention weight during training
        if self.training:
            del attention

        # x: (batch_size, input_length1, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # x: (batch_size, input_length1, d_model)
        x = self.WO(x)

        if self.training:
            return x,
        else:
            return x, attention


class RelativePosition(nn.Module):

    def __init__(self, d_a, max_relative_position, max_cache_len):
        super(RelativePosition, self).__init__()
        self.d_a = d_a
        self.max_relative_position = max_relative_position
        self.max_cache_seq_len = max_cache_len
        self.embedding_table = nn.Embedding(2 * max_relative_position + 1, d_a)
        self.register_buffer("ref_pos", self.create_ref_pos(max_relative_position, max_cache_len), persistent=False)

    def forward(self, length_q, length_k):
        if max(length_k, length_q) > self.max_cache_seq_len:
            self.max_cache_seq_len = max(length_k, length_q)
            self.ref_pos = self.create_ref_pos(self.max_relative_position, self.max_cache_seq_len)

        current_ref_pos = self.ref_pos.narrow(0, 0, length_q).narrow(1, 0, length_k)
        return self.embedding_table(current_ref_pos)

    def create_ref_pos(self, max_relative_position, max_cache_seq_len):
        _rpm = torch.arange(-max_cache_seq_len + 1, 1, dtype=torch.long,
                            device=self.embedding_table.weight.device).unsqueeze(0)
        ref_pos = (_rpm - _rpm.t()).clamp(min=-max_relative_position, max=max_relative_position) + max_relative_position
        return ref_pos
