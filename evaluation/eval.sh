python -m evaluation.eval \
    --device cuda:2 \
    --load /data/rrjin/NMT/data/model__9_252493.382682 \
    --test_src_path /data/rrjin/NMT/data/test_src.spa \
    --test_tgt_path /data/rrjin/NMT/data/test_tgt.en \
    --test_src_vocab_path /data/rrjin/NMT/data/src.spa.vocab \
    --test_tgt_vocab_path /data/rrjin/NMT/data/tgt.en.vocab \
    --translation_output /data/rrjin/NMT/data/translations.txt