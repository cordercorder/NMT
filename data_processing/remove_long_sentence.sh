# deprecated

cd ..

python -m data_processing.remove_long_sentence \
    --src_file_path_list /data/rrjin/NMT/data/ted_data/corpus/shuffled_train_data_src_tok_joint_bpe_32000.combine\
    --tgt_file_path_list /data/rrjin/NMT/data/ted_data/corpus/shuffled_train_data_tgt_tok_joint_bpe_32000.en \
    --max_sentence_length 100