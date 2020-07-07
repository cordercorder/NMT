python -u -m evaluation.eval \
    --device cuda:0 \
    --load /data/rrjin/NMT/tmpdata/model_basic_multi_gpu_lstm7__0_0.490599 \
    --test_src_path /data/rrjin/NMT/tmpdata/test_src.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/test_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --translation_output /data/rrjin/NMT/tmpdata/beam_translations_basic_multi_gpu_lstm9.txt