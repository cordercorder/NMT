cd ..

python -u -m data_processing.shuffle_parallel_corpus \
    --seed 998244353 \
    --src_file_path /data/rrjin/NMT/data/ted_data/corpus/train_data_src_joint_bpe_32000_max_len_100.combine \
    --tgt_file_path /data/rrjin/NMT/data/ted_data/corpus/train_data_tgt_joint_bpe_32000_max_len_100.en \
    --shuffled_src_file_path /data/rrjin/NMT/data/ted_data/corpus/train_data_src_joint_bpe_32000_max_len_100_shuffled.combine \
    --shuffled_tgt_file_path /data/rrjin/NMT/data/ted_data/corpus/train_data_tgt_joint_bpe_32000_max_len_100_shuffled.en