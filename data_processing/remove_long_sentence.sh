cd ..

python -m data_processing.remove_long_sentence \
    --src_file_path_list /data/rrjin/NMT/data/bible_data/corpus/train_src_combine_joint_bpe_22000.txt\
    --tgt_file_path_list /data/rrjin/NMT/data/bible_data/corpus/train_tgt_en_joint_bpe_22000.txt \
    --max_sentence_length 100