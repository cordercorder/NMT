class Hypothesis:

    def __init__(self, decoder_hidden_state=None, decoder_output=None):

        self.decoder_hidden_state = decoder_hidden_state
        self.decoder_output = decoder_output
        self.score = 0
        self.pred_index_list = []
        self.complete = False

    def __len__(self):
        return len(self.pred_index_list)
