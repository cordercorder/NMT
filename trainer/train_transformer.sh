cd ..

python -m trainer.train_transformer \
    --device cuda:2 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --src_language spa \
    --tgt_language en \
    --src_path /data/rrjin/NMT/tmpdata/train_src_tok.spa \
    --tgt_path /data/rrjin/NMT/tmpdata/train_tgt_tok.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --checkpoint /data/rrjin/NMT/tmpdata/test_transformer \
    --dropout 0.1 \
    --rebuild_vocab \
    --learning_rate 0.00005