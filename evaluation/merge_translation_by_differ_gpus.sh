cd ..

python -u -m evaluation.merge_translation_by_differ_gpus \
    --multi_gpu_translation_dir /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_greedy \
    --is_tok \
    --merged_translation_dir /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_greedy_merged