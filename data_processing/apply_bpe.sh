python -u apply_bpe.py \
    --train_file_tgt_path /data/rrjin/NMT/data/ted_data/corpus/train_data_src_filter1.combine \
    --train_file_src_path  /data/rrjin/NMT/data/ted_data/corpus/train_data_tgt_filter1.en \
    --num_operations 32000 \
    --vocabulary_threshold 0 \
    --file_tgt_path_list /data/rrjin/NMT/data/ted_data/corpus/train_data_src_filter1.combine \
                         /data/rrjin/NMT/data/ted_data/corpus/test_data_src.combine \
                         /data/rrjin/NMT/data/ted_data/corpus/dev_data_src.combine \
    --file_src_path_list /data/rrjin/NMT/data/ted_data/corpus/train_data_tgt_filter1.en \
                         /data/rrjin/NMT/data/ted_data/corpus/test_data_tgt.en \
                         /data/rrjin/NMT/data/ted_data/corpus/dev_data_tgt.en \
    --output_directory /data/rrjin/NMT/data/ted_data/corpus/tmp
