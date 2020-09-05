cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/train_data_src_filter1_joint_bpe_32000.combine \
    --tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/train_data_tgt_filter1_joint_bpe_32000.en \
    --output_src_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/train_data_src_joint_bpe_32000_max_len_100.combine \
    --output_tgt_file_path_list \
        /data/rrjin/NMT/data/ted_data/corpus/train_data_tgt_joint_bpe_32000_max_len_100.en \
    --operation remove_long_sentence \
    --max_sentence_length 100