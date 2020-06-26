import torch
import torch.nn as nn


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

        return output, hidden_state

    def forward(self, input_batch, target_batch, encoder_hidden_state, use_teacher_forcing):
        """
        :param input_batch: the shape of input_batch is (input_length, batch_size). type: Tensor
        :param target_batch: the shape of target_batch is (input_length, batch_size). type: Tensor
        :param encoder_hidden_state: the shape of encoder_hidden_state is (num_layers, batch_size,hidden_size).
                                     type: Tensor
        :param use_teacher_forcing: whether use teacher forcing or not. type: bool
        """

        decoder_hidden_state = encoder_hidden_state

        # decoder_input: (1, batch_size)
        decoder_input = input_batch[0].view(1, -1)

        # decoder_output_list: (input_length, batch_size, vocab_size)
        decoder_batch_output = torch.zeros(size=(target_batch.size(0), target_batch.size(1), self.vocab_size),
                                           device=input_batch.device)

        for i in range(1, target_batch.size(0)-1):
            # target_batch[i]: (batch_size, )
            decoder_output, decoder_hidden_state = self.decode_batch(decoder_input, decoder_hidden_state)

            if use_teacher_forcing:
                decoder_input = target_batch[i].view(1, -1)
            else:
                # pred_tensor: (1, batch_size, 1)
                # pred_index: (1, batch_size, 1)
                pred_tensor, pred_index = decoder_output.topk(1, dim=2)
                decoder_input = torch.squeeze(pred_index, 2)

            decoder_batch_output[i] = decoder_output[0]

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
            # encoder_hidden_state: (num_layers, 2, batch_size, hidden_size)
            encoder_hidden_state = encoder_hidden_state.view(-1, 2, encoder_hidden_state.size(1),
                                                             encoder_hidden_state.size(2))
            # encoder_hidden_state: (num_layers, 2, batch_size, 2 * hidden_size)
            encoder_hidden_state = torch.cat([encoder_hidden_state[:, 0, :, :], encoder_hidden_state[:, 1, :, :]],
                                             dim=2)
        decoder_output = self.decoder(input_batch, target_batch, encoder_hidden_state, use_teacher_forcing)

        return decoder_output