python -u apply_bpe.py \
    --train_file_src_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_filter1.en \
    --train_file_tgt_path  /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_filter1.combine \
    --num_operations 32000 \
    --vocabulary_threshold 0 \
    --file_src_path_list /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_filter1.en \
                         /data/rrjin/NMT/data/ted_data/sub_corpus/data/test_data.en \
                         /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data.en \
    --file_tgt_path_list /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_filter1.combine \
                         /data/rrjin/NMT/data/ted_data/sub_corpus/data/test_data.combine \
                         /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data.combine \
    --output_directory /data/rrjin/NMT/data/ted_data/sub_corpus/data
