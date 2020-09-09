cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/dev_data_src_joint_bpe_32000_max_len_150.combine \
    --tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/dev_data_tgt_max_len_150.en \
    --output_src_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/dev_data_src_joint_bpe_32000_max_len_150_sorted.combine \
    --output_tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/dev_data_tgt_max_len_150_sorted.en \
    --operation sort_sentence
