cd ..

python -u -m data_processing.parallel_corpus_manager \
    --src_file_path_list \
        /data/cordercorder/NMT/data/corpus/clean.tok.pe.32000 \
    --tgt_file_path_list \
        /data/cordercorder/NMT/data/corpus/clean.tok.zh.32000 \
    --output_src_file_path_list \
        /data/cordercorder/NMT/data/corpus/clean.tok.pe_max_len_80.32000 \
    --output_tgt_file_path_list \
        /data/cordercorder/NMT/data/corpus/clean.tok.zh_max_len_80.32000 \
    --operation remove_long_sentence \
    --max_sentence_length 80
