python -u apply_bpe.py \
    --train_file_src_path /data/rrjin/NMT/data/ted_data/shuffled_train_data_src_tok.combine \
    --train_file_tgt_path  /data/rrjin/NMT/data/ted_data/shuffled_train_data_tgt_tok.en \
    --num_operations 20000 \
    --vocabulary_threshold 0 \
    --file_src_path_list /data/rrjin/NMT/data/ted_data/shuffled_train_data_src_tok.combine \
                         /data/rrjin/NMT/data/ted_data/raw_test_data_src_tok.combine \
                         /data/rrjin/NMT/data/ted_data/raw_dev_data_src_tok.combine \
    --file_tgt_path_list /data/rrjin/NMT/data/ted_data/shuffled_train_data_tgt_tok.en \
                         /data/rrjin/NMT/data/ted_data/raw_test_data_tgt_tok.en \
                         /data/rrjin/NMT/data/ted_data/raw_dev_data_tgt_tok.en \
    --output_directory /data/rrjin/NMT/data/ted_data
