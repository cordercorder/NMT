python -m trainer.train \
    --device cpu \
    --src_language en \
    --tgt_language en \
    --src_path /cordercorder/NMT/data/src.en \
    --tgt_path /cordercorder/NMT/data/tgt.en \
    --src_vocab_path /cordercorder/NMT/data/src.en.vocab \
    --tgt_vocab_path /cordercorder/NMT/data/tgt.en.vocab \
    --rnn_type rnn \
    --embedding_size 128 \
    --hidden_size 128 \
    --num_layers 3 \
    --checkpoint /cordercorder/NMT/data/model \
    --batch_size 1