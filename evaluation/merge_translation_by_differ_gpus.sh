cd ..

multi_gpu_translation_dir=/data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy

python -u -m evaluation.merge_translation_by_differ_gpus \
    --multi_gpu_translation_dir ${multi_gpu_translation_dir} \
    --is_tok \
    --merged_translation_dir ${multi_gpu_translation_dir}_merged
