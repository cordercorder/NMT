cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_src_tok_joint_bpe_32000.combine \
    --tgt_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_tgt_tok_joint_bpe_32000.en \
    --output_src_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_src_tok_joint_bpe_32000_filtered.combine \
    --output_tgt_file_path_list /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_tgt_tok_joint_bpe_32000_filtered.en \
    --operation remove_long_sentence \
    --max_sentence_length 100