cd ..

python -u -m data_processing.tok_data \
    --data_path /data/rrjin/NMT/data/ted_data/shuffled_train_data_src.combine \
                /data/rrjin/NMT/data/ted_data/raw_dev_data_src.combine \
                /data/rrjin/NMT/data/ted_data/raw_test_data_src.combine \
    --do_lower_case