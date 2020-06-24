import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, num_layers, dropout_=0, bidirectional_=False):

        """
        :param rnn_type: type of rnn. supporting (RNN, LSTM, GRU). type: str
        :param vocab_size: size of vocabulary. type: int
        :param embedding_size: size of each embedding vector
        :param hidden_size: hidden size in each rnn layers. type: int
        :param num_layers: number of rnn layers. type: int
        :param dropout_: if non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer,
                         with dropout probability equal to dropout. Default: 0. type: float
        :param bidirectional_: if True, becomes a bidirectional RNN. Default: False. type: bool
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

    def forward(self, input_batch, input_length, padding_value_):

        """
        :param input_batch: the shape of input_batch is (input_length, batch_size). type: Tensor
        :param input_length: length of sequence in the batch. type: list or Tensor
        :param padding_value_: value for padded element. type: float
        """

        # input_batch: (input_length, batch_size, embedding_size)
        input_batch = self.embedding(input_batch)

        input_batch = nn.utils.rnn.pack_padded_sequence(input_batch, input_length, enforce_sorted=False)

        # output: (input_length, batch_size, num_directions * hidden_size)
        # hidden_state is hn or (hn, cn)
        # hn: (num_layers * num_directions, batch_size, hidden_size)
        # cn: (num_layers * num_directions, batch_size, hidden_size)
        output, hidden_state = self.rnn(input_batch)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, padding_value=padding_value_)

        return output, hidden_state


class Decoder(nn.Module):

    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, num_layers, dropout_=0):

        """
        :param rnn_type: type of rnn. supporting (RNN, LSTM, GRU). type: str
        :param vocab_size: size of vocabulary. type: int
        :param embedding_size: size of each embedding vector
        :param hidden_size: hidden size in each rnn layers. type: int
        :param num_layers: number of rnn layers. type: int
        :param dropout_: if non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer,
                         with dropout probability equal to dropout. Default: 0. type: float
        """

        super(Decoder, self).__init__()

        rnn_type = rnn_type.lower().strip()

        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out_ = dropout_

        if rnn_type != "gru" and rnn_type != "rnn" and rnn_type != "lstm":
            raise Exception("Unknown type of rnn")

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        if rnn_type == "gru":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout_)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout_)
        else:
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_)

        self.fc = nn.Linear(self.hidden_size, vocab_size)

    def decode_batch(self, decoder_input, decoder_hidden_state):
        """
        :param decoder_input:
        :param decoder_hidden_state:
        """
        # decoder_input: (1, batch_size, embedding_size)
        decoder_input = self.embedding(decoder_input)

        # output: (1, batch_size, hidden_size)
        # hidden_state is hn or (hn, cn)
        # hn: (num_layers, batch_size, hidden_size)
        # cn: (num_layers, batch_size, hidden_size)
        output, hidden_state = self.rnn(decoder_input, decoder_hidden_state)

        # output: (1, batch_size, vocab_size)
        output = self.fc(output)

        # output: (1, batch_size, vocab_size)
        pred = torch.nn.functional.log_softmax(output, dim=2)

        return pred, hidden_state

    def forward(self, input_batch, encoder_hidden_state=None, teacher_forcing_ratio=0):
        """
        :param input_batch: the shape of input_batch is (input_length, batch_size). type: Tensor
        :param encoder_hidden_state: the shape of encoder_hidden_state is (num_layers, batch_size,hidden_size).
                                     type: Tensor
        :param teacher_forcing_ratio: probability of using teacher forcing
        """

        decoder_hidden_state = encoder_hidden_state

        # decoder_input: (1, batch_size)
        decoder_input = input_batch[0].view(1, -1)

        for i in range(1, input_batch.size(0)):

            decoder_input = input_batch[i]





