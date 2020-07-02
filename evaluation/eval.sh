python -m evaluation.eval \
    --device cuda:2 \
    --load /data/rrjin/NMT/tmpdata/model_fix_gru__9_0.009219 \
    --test_src_path /data/rrjin/NMT/tmpdata/test_src.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/test_tgt.en \
    --test_src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --test_tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --translation_output /data/rrjin/NMT/tmpdata/translations_gru.txt