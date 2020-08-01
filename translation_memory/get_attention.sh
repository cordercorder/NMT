cd ..

python -m translation_memory.get_attention \
    --device cpu \
    --load /data/rrjin/NMT/tmpdata/test_transformer_multi_gpu3_rank0 \
    --test_src_path /data/rrjin/NMT/tmpdata/val_src_tok.spa \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --picture_directory /data/rrjin/NMT/tmpdata