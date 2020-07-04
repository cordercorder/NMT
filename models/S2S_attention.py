import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size1, hidden_size2, attention_size):

        """
        :param hidden_size1: num_directions * hidden_size
        :param hidden_size2: num_layers of decoder * hidden_size of decoder
        :param attention_size: any integer
        """

        super(BahdanauAttention, self).__init__()

        self.W1 = nn.Linear(hidden_size1, attention_size)
        self.W2 = nn.Linear(hidden_size2, attention_size)

        self.V = nn.Linear(attention_size, 1)


    def forward(self, hs, ht):

        """
        :param hs: last hidden state of decoder. (num_layers, batch_size, hidden_size)
        :param ht: encoder output. (input_length, batch_size, num_directions * hidden_size)
        :return: context_vector, attention_weights
        """

        # ht: (batch_size, input_length, num_directions * hidden_size)
        ht = torch.transpose(ht, 0, 1)

        # hs: (batch_size, num_layers, hidden_size)
        hs = torch.transpose(hs, 0, 1)

        # hs: (batch_size, 1, num_layers * hidden_size)
        hs = hs.reshape(hs.size(0), 1, -1)

        # W1@ht: (batch_size, input_length, attention_size)
        # W2@hs: (batch_size, 1, attention_size)

        # W1@ht + W2@hs: (batch_size, input_length, attention_size)

        # score: (batch_size, input_length, 1)
        score = self.V(torch.tanh(self.W1(ht) + self.W2(hs)))

        # attention_weight: (batch_size, input_length, 1)
        attention_weights = F.softmax(score, dim=1)

        # context_vector: (batch_size, input_length, num_directions * hidden_size)
        context_vector = attention_weights * ht

        # context_vector: (batch_size, num_directions * hidden_size)
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class Encoder(nn.Module):

    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, num_layers, dropout_=0, bidirectional_=True):

        """
        :param rnn_type: type of rnn. supporting (RNN, LSTM, GRU). type: str
        :param vocab_size: size of vocabulary. type: int
        :param embedding_size: size of each embedding vector
        :param hidden_size: hidden size in each rnn layers. type: int
        :param num_layers: number of rnn layers. type: int
        :param dropout_: if non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer,
                         with dropout probability equal to dropout. Default: 0. type: float
        :param bidirectional_: if True, becomes a bidirectional RNN. Default: True. type: bool
        """

        super(Encoder, self).__init__()

        rnn_type = rnn_type.lower().strip()

        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out_ = dropout_
        self.bidirectional_ = bidirectional_

        if rnn_type != "gru" and rnn_type != "rnn" and rnn_type != "lstm":
            raise Exception("Unknown type of rnn")

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        if rnn_type == "gru":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout_, bidirectional=bidirectional_)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout_, bidirectional=bidirectional_)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_, bidirectional=bidirectional_)

    def forward(self, input_batch):

        """
        :param input_batch: the shape of input_batch is (input_length, batch_size). type: Tensor
        """

        # input_batch: (input_length, batch_size, embedding_size)
        input_batch = self.embedding(input_batch)

        # output: (input_length, batch_size, num_directions * hidden_size)
        # hidden_state is hn or (hn, cn)
        # hn: (num_layers * num_directions, batch_size, hidden_size)
        # cn: (num_layers * num_directions, batch_size, hidden_size)
        output, hidden_state = self.rnn(input_batch)

        return output, hidden_state


class AttentionDecoder(nn.Module):

    def __init__(self, rnn_type, vocab_size, embedding_size, input_size, hidden_size, num_layers, attention,
                 dropout_=0):

        """
        :param rnn_type: type of rnn. supporting (RNN, LSTM, GRU). type: str
        :param vocab_size: size of vocabulary. type: int
        :param embedding_size: size of each embedding vector
        :param input_size: input_size of rnn
        :param hidden_size: hidden size in each rnn layers. type: int
        :param num_layers: number of rnn layers. type: int
        :param attention: attention layer
        :param dropout_: if non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer,
                         with dropout probability equal to dropout. Default: 0. type: float
        """

        super(AttentionDecoder, self).__init__()

        rnn_type = rnn_type.lower().strip()

        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.drop_out_ = dropout_

        if rnn_type != "gru" and rnn_type != "rnn" and rnn_type != "lstm":
            raise Exception("Unknown type of rnn")

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        if rnn_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout_)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout_)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_)

        self.fc = nn.Linear(self.hidden_size, vocab_size)

    def decode_batch(self, decoder_input, decoder_hidden_state, encoder_output, return_attention_weight=False):
        """
        :param decoder_input: (1, batch_size)
        :param decoder_hidden_state: (num_layers, batch_size, hidden_size)
        :param encoder_output: (input_length, batch_size, num_directions * hidden_size)
        :param return_attention_weight: whether return attention weight or not. default False
        """
        # decoder_input: (1, batch_size, embedding_size)
        decoder_input = self.embedding(decoder_input)

        if isinstance(decoder_hidden_state, tuple):

            hn, cn = decoder_hidden_state

            # atten_decoder_hidden_state: (num_layers, batch_size, hidden_size)
            atten_decoder_hidden_state = hn

        else:

            # atten_decoder_hidden_state: (num_layers, batch_size, hidden_size)
            atten_decoder_hidden_state = decoder_hidden_state

        # context_vector: (batch_size, num_directions * hidden_size)
        # attention_weight: (batch_size, input_length, 1)
        context_vector, attention_weights = self.attention(atten_decoder_hidden_state, encoder_output)

        # context_vector: (1, batch_size, num_directions * hidden_size)
        context_vector = context_vector.view(1, context_vector.size(0), context_vector.size(1))

        # decoder_input: (1, batch_size, embedding_size + num_directions * hidden_size)
        decoder_input = torch.cat([decoder_input, context_vector], dim=2)

        # output: (1, batch_size, hidden_size)
        # hidden_state is hn or (hn, cn)
        # hn: (num_layers, batch_size, hidden_size)
        # cn: (num_layers, batch_size, hidden_size)
        output, hidden_state = self.rnn(decoder_input, decoder_hidden_state)

        # output: (1, batch_size, vocab_size)
        output = self.fc(output)

        if return_attention_weight:
            return output, hidden_state, attention_weights

        return output, hidden_state

    def forward(self, input_batch, target_batch, encoder_hidden_state, encoder_output, use_teacher_forcing):
        """
        :param input_batch: the shape of input_batch is (input_length, batch_size). type: Tensor
        :param target_batch: the shape of target_batch is (input_length, batch_size). type: Tensor
        :param encoder_hidden_state: the shape of encoder_hidden_state is (num_layers, batch_size,hidden_size).
                                     type: Tensor
        :param encoder_output: the shape of encoder_output is (input_length, batch_size, num_directions * hidden_size)
        :param use_teacher_forcing: whether use teacher forcing or not. type: bool
        """

        decoder_hidden_state = encoder_hidden_state

        # decoder_input: (1, batch_size)
        decoder_input = input_batch[0].view(1, -1)

        # decoder_batch_output: type: list. (input_length-1, batch_size, vocab_size)
        decoder_batch_output = []

        for i in range(1, target_batch.size(0)):

            # target_batch[i]: (batch_size, )
            decoder_output, decoder_hidden_state = self.decode_batch(decoder_input, decoder_hidden_state, encoder_output)

            if use_teacher_forcing:
                decoder_input = target_batch[i].view(1, -1)
            else:
                # pred_tensor: (1, batch_size, 1)
                # pred_index: (1, batch_size, 1)
                # pred_tensor, pred_index = decoder_output.topk(1, dim=2)
                # decoder_input = torch.squeeze(pred_index, 2)

                pred_index = torch.argmax(decoder_output, dim=2)
                decoder_input = pred_index

            decoder_batch_output.append(decoder_output[0])

        return decoder_batch_output


class S2S(nn.Module):

    def __init__(self, encoder, decoder):

        """
        :param encoder: Encoder
        :param decoder: Decoder
        """

        super(S2S, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_batch, target_batch, use_teacher_forcing):
        """
        :param input_batch: the shape of input_batch is (input_length, batch_size). type: Tensor
        :param target_batch: the shape of target_batch is (input_length, batch_size). type: Tensor
        :param use_teacher_forcing: whether use teacher forcing or not. type: bool
        :return: output of decoder, shape: (input_length, batch_size, vocab_size). type: Tensor
        """

        # encoder_output: (input_length, batch_size, num_directions * hidden_size)
        # encoder_hidden_state: (num_layers * num_directions, batch_size, hidden_size)
        encoder_output, encoder_hidden_state = self.encoder(input_batch)

        if self.encoder.bidirectional_:

            if self.encoder.rnn_type == "lstm":
                # hn: (num_layers * num_directions, batch_size, hidden_size)
                # cn: (num_layers * num_directions, batch_size, hidden_size)
                hn, cn = encoder_hidden_state

                # hn: (num_layers, 2, batch_size, hidden_size)
                # cn: (num_layers, 2, batch_size, hidden_size)
                hn = hn.view(-1, 2, hn.size(1), hn.size(2))
                cn = cn.view(-1, 2, cn.size(1), cn.size(2))

                # hn: (num_layers, batch_size, 2 * hidden_size)
                # cn: (num_layers, batch_size, 2 * hidden_size)
                hn = torch.cat([hn[:, 0, :, :], hn[:, 1, :, :]], dim=2)
                cn = torch.cat([cn[:, 0, :, :], cn[:, 1, :, :]], dim=2)
                encoder_hidden_state = (hn, cn)

            else:
                # encoder_hidden_state: (num_layers, 2, batch_size, hidden_size)
                encoder_hidden_state = encoder_hidden_state.view(-1, 2, encoder_hidden_state.size(1),
                                                                 encoder_hidden_state.size(2))
                # encoder_hidden_state: (num_layers, batch_size, 2 * hidden_size)
                encoder_hidden_state = torch.cat([encoder_hidden_state[:, 0, :, :], encoder_hidden_state[:, 1, :, :]],
                                                 dim=2)
        decoder_output = self.decoder(input_batch, target_batch, encoder_hidden_state, encoder_output,
                                      use_teacher_forcing)

        return decoder_output

    def train_batch(self, input_batch, target_batch, padding_value, criterion, use_teacher_forcing):

        output = self(input_batch, target_batch, use_teacher_forcing)

        batch_loss = torch.zeros(1, device=input_batch.device)

        for k in range(1, target_batch.size(0)):
            # tmp_output_batch: (batch_size, vocab_size)
            # tmp_target_batch: (batch_size, )
            tmp_output_batch = output[k - 1]
            tmp_target_batch = target_batch[k]

            mask = torch.ne(tmp_target_batch, padding_value).float()
            # tmp_loss: (batch_size, )
            tmp_loss = criterion(tmp_output_batch, tmp_target_batch)

            tmp_loss *= mask

            batch_loss += torch.sum(tmp_loss)

        return batch_loss