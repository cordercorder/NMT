python -u train_transformer.py \
    --device cuda:0 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 512 \
    --src_language spa \
    --tgt_language en \
    --src_path /data/rrjin/NMT/tmpdata/train_src.spa \
    --tgt_path /data/rrjin/NMT/tmpdata/train_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src2.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt2.en.vocab \
    --checkpoint /data/rrjin/NMT/tmpdata/test_transformer \
    --learning_rate 0.0005 \
    --normalize \
    --sort_sentence_by_length \
    --dropout 0.1 \
    --rebuild_vocab