cd ..

python -m trainer.train \
    --device cuda:3 \
    --src_language spa \
    --tgt_language en \
    --src_path /data/rrjin/NMT/tmpdata/train_src_tok.spa \
    --tgt_path /data/rrjin/NMT/tmpdata/train_tgt_tok.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --rnn_type lstm \
    --embedding_size 512 \
    --hidden_size 512 \
    --num_layers 3 \
    --checkpoint /data/rrjin/NMT/tmpdata/test_basic \
    --batch_size 32 \
    --dropout 0.1 \
    --rebuild_vocab \
    --attention_size 512 \
    --sort_sentence_by_length