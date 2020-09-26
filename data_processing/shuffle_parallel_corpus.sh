cd ..

python -u -m data_processing.shuffle_parallel_corpus \
    --seed 998244353 \
    --src_file_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_joint_bpe_32000_max_len_100.en \
    --tgt_file_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_joint_bpe_32000_max_len_100.combine \
    --shuffled_src_file_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_joint_bpe_32000_max_len_100_shuffled.en \
    --shuffled_tgt_file_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_joint_bpe_32000_max_len_100_shuffled.combine
