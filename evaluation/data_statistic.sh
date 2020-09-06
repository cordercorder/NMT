cd ..

python -u -m evaluation.data_statistic \
    --data_path /data/rrjin/NMT/data/ted_data/corpus/train_data_src_joint_bpe_32000_max_len_100_shuffled.combine \
    --picture_path /data/rrjin/NMT/data/ted_data/corpus/train_data_statistic.jpg \
    --language_data /data/rrjin/NMT/data/ted_data/ted_lang_tok_to_languages.json