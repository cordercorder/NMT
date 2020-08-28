from models.transformer import Encoder


class LangEncoder(Encoder):

    def __init__(self, src_vocab_size, max_src_len, d_model, num_layers, num_heads, d_ff, dropout, device):
        super().__init__(src_vocab_size, max_src_len, d_model, num_layers, num_heads, d_ff, dropout, device)

    def forward(self, src, src_mask):
        # src: (batch_size, input_length)
        # src_mask: (batch_size, 1, 1, src_length)

        # src: (batch_size, input_length, embedding_size)
        src = self.token_embedding(src)

        # lang_embedding: (batch_size, 1, embedding_size)
        lang_embedding = src[:, 1]
        lang_embedding = lang_embedding.unsqueeze(1)

        src = src + lang_embedding
        del lang_embedding

        src = src * self.scale
        src = self.dropout(self.pos_embedding(src))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
