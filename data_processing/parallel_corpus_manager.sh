cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_src_tok_joint_bpe_32000_sorted_filter1.combine \
                         /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_src_tok_joint_bpe_32000_sorted_filter1.combine \
    --tgt_file_path_list /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_tgt_tok_sorted_filter1.en \
                         /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_tgt_tok_sorted_filter1.en \
    --output_src_file_path_list /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_src_tok_joint_bpe_32000_sorted_filter2.combine \
                                /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_src_tok_joint_bpe_32000_sorted_filter2.combine \
    --output_tgt_file_path_list /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_tgt_tok_sorted_filter2.en \
                                /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_tgt_tok_sorted_filter2.en \
    --operation remove_long_sentence \
    --max_sentence_length 200