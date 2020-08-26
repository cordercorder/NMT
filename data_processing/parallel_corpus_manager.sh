cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_src_tok_joint_bpe_32000_filtered2.combine \
    --tgt_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_tgt_tok_filtered2.en \
    --output_src_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_src_tok_joint_bpe_32000_filtered2_sorted.combine \
    --output_tgt_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_tgt_tok_filtered2_sorted.en \
    --operation sort_sentence