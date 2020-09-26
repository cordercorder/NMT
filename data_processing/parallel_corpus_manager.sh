cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_filter1_joint_bpe_32000.en \
    --tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_filter1_joint_bpe_32000.combine \
    --output_src_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_joint_bpe_32000_max_len_100.en \
    --output_tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_joint_bpe_32000_max_len_100.combine \
    --operation remove_long_sentence \
    --max_sentence_length 100
