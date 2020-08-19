cd ..

python -u -m data_processing.tok_data \
    --data_path /data/rrjin/NMT/data/ted_data/corpus/shuffled_train_data_src.combine \
                /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_src.combine \
                /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_src.combine \
                /data/rrjin/NMT/data/ted_data/corpus/shuffled_train_data_tgt.en \
                /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_tgt.en \
                /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_tgt.en \
    --do_lower_case