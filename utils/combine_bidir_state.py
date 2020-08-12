import torch


def combine_bidir_hidden_state(s2s, encoder_hidden_state: torch.tensor):

    if s2s.encoder.bidirectional_:

        if s2s.encoder.rnn_type == "lstm":

            # hn: (num_layers * num_directions, batch_size, hidden_size)
            # cn: (num_layers * num_directions, batch_size, hidden_size)
            hn, cn = encoder_hidden_state

            # hn: (num_layers, batch_size, num_directions * hidden_size)
            hn = hn.view(-1, 2, hn.size(1), hn.size(2))
            hn = torch.cat([hn[:, 0, :, :], hn[:, 1, :, :]], dim=2)

            # cn: (num_layers, batch_size, num_directions * hidden_size)
            cn = cn.view(-1, 2, cn.size(1), cn.size(2))
            cn = torch.cat([cn[:, 0, :, :], cn[:, 1, :, :]], dim=2)
            encoder_hidden_state = (hn, cn)

        else:
            encoder_hidden_state = encoder_hidden_state.view(-1, 2, encoder_hidden_state.size(1),
                                                             encoder_hidden_state.size(2))

            # decoder_hidden_state: (num_layers, batch_size, num_directions * hidden_size)
            encoder_hidden_state = torch.cat([encoder_hidden_state[:, 0, :, :],
                                              encoder_hidden_state[:, 1, :, :]], dim=2)
    return encoder_hidden_state
