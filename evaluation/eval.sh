python -m evaluation.eval \
    --device cpu \
    --load /cordercorder/NMT/data/model__9_9.204300 \
    --test_src_path /cordercorder/NMT/data/src.en \
    --test_tgt_path /cordercorder/NMT/data/tgt.en \
    --test_src_vocab_path /cordercorder/NMT/data/src.en.vocab \
    --test_tgt_vocab_path /cordercorder/NMT/data/tgt.en.vocab