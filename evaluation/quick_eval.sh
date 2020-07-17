python -u -m evaluation.quick_eval \
    --device cuda:0 \
    --model_prefix /data/rrjin/NMT/tmpdata/test_transformer__9_5.188601 \
    --test_src_path /data/rrjin/NMT/tmpdata/toy_data_src.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/toy_data_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src2.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt2.en.vocab \
    --translation_output /data/rrjin/NMT/tmpdata/tmp \
    --transformer