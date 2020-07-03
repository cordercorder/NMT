python -m evaluation.eval \
    --device cuda:3 \
    --load /data/rrjin/NMT/tmpdata/model_attention_lstm__9_0.259051 \
    --test_src_path /data/rrjin/NMT/tmpdata/val_src.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/val_tgt.en \
    --test_src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --test_tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --translation_output /data/rrjin/NMT/tmpdata/beam_translations_val_lstm.txt