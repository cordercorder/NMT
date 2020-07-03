class Hypothesis:

    def __init__(self, decoder_hidden_state=None, decoder_input=None, prev_hypothesis=None):

        self.decoder_hidden_state = decoder_hidden_state
        self.decoder_input = decoder_input
        self.score = 0
        self.prev_hypothesis = prev_hypothesis
        self.pred_index_list = []

    def __len__(self):
        return len(self.pred_index_list)
