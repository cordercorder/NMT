cd ..

multi_gpu_translation_dir=/data/rrjin/NMT/data/ted_data/translation/transformer_test_beam_size_5
merged_translation_dir=${multi_gpu_translation_dir}_merged

python -u -m evaluation.merge_translation_by_differ_gpus \
    --multi_gpu_translation_dir ${multi_gpu_translation_dir} \
    --is_tok \
    --merged_translation_dir ${merged_translation_dir}

