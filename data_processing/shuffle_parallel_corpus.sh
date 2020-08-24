cd ..

python -u -m data_processing.shuffle_parallel_corpus \
    --seed 998244353 \
    --src_file_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_train_data_src.combine \
    --tgt_file_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_train_data_tgt.en \
    --shuffled_src_file_path /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_src.combine \
    --shuffled_tgt_file_path /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_tgt.en