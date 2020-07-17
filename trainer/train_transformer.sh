python -u train_transformer.py \
    --device cuda:2 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --src_language spa \
    --tgt_language en \
    --src_path /data/rrjin/NMT/tmpdata/train_src.spa \
    --tgt_path /data/rrjin/NMT/tmpdata/train_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src2.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt2.en.vocab \
    --checkpoint /data/rrjin/NMT/tmpdata/test_transformer \
    --normalize \
    --dropout 0.1 \
    --rebuild_vocab \
    --learning_rate 0.00001