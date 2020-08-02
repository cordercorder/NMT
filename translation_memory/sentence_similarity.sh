cd ..

python -m translation_memory.sentence_similarity \
    --device cuda:2 \
    --load /data/rrjin/NMT/tmpdata/test_transformer_multi_gpu3_rank0 \
    --test_src_path /data/rrjin/NMT/tmpdata/similarity_test_src.spa \
    --src_memory_path /data/rrjin/NMT/tmpdata/src_memory.spa \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --test_src_with_memory_path /data/rrjin/NMT/tmpdata/src_with_memory.txt
