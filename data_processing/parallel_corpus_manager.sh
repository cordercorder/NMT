cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data_joint_bpe_32000_max_len_150.en \
    --tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data_max_len_150.combine \
    --output_src_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data_joint_bpe_32000_max_len_150_sorted.en \
    --output_tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data_max_len_150_sorted.combine \
    --operation sort_sentence
