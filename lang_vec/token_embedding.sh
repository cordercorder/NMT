cd ..

python -m lang_vec.token_embedding \
    --device cuda:0 \
    --load /data/rrjin/NMT/data/models/basic_multi_gpu_lstm__9_3.178944 \
    --src_vocab_path /data/rrjin/NMT/data/src_combine_32000.vocab \
    --lang_token_list_path /data/rrjin/NMT/data/bible_lang_tok \
    --lang_vec_path /data/rrjin/NMT/data/bible_lang_vec_basic_rnn.token_embedding