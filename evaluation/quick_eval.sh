cd ..

python -u -m evaluation.quick_eval \
    --device cuda:3 \
    --model_prefix /data/rrjin/NMT/tmpdata/test_transformer_multi_gpu3_rank \
    --test_src_path /data/rrjin/NMT/tmpdata/test_src_tok.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/test_tgt_tok.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src2.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt2.en.vocab \
    --translation_output /data/rrjin/NMT/tmpdata/test_tmp \
    --transformer