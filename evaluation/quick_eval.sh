cd ..

python -u -m evaluation.quick_eval \
    --device cuda:0 \
    --model_load /data/rrjin/NMT/tmpdata/test_transformer_multi_gpu3_rank0 \
    --test_src_path /data/rrjin/NMT/tmpdata/test_src_tok.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/test_tgt_tok.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src2.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt2.en.vocab \
    --translation_output_dir /data/rrjin/NMT/tmpdata/ \
    --transformer \
    --beam_size 3 \
    --record_time