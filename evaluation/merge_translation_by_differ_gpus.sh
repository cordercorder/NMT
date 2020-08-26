cd ..

python -u -m evaluation.merge_translation_by_differ_gpus \
    --multi_gpu_translation_dir /data/rrjin/NMT/data/ted_data/translation/transformer_test \
    --is_tok \
    --merged_translation_dir /data/rrjin/NMT/data/ted_data/translation