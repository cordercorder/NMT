python -u -m evaluation.quick_eval \
    --device cuda:0 \
    --model_prefix /data/rrjin/NMT/tmpdata/test_basic__9_0.639688 \
    --test_src_path /data/rrjin/NMT/tmpdata/test_src.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/test_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --translation_output /data/rrjin/NMT/tmpdata/tmp