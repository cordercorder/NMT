python -m evaluation.draw_attention \
    --device cuda:1 \
    --load /data/rrjin/NMT/tmpdata/model_attention_lstm__9_0.272557 \
    --val_src_path /data/rrjin/NMT/tmpdata/validate_attention_src.spa \
    --val_tgt_path /data/rrjin/NMT/tmpdata/validate_attention_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --picture_path /data/rrjin/NMT/tmpdata/attention_plot