cd ..

python -m lang_vec.token_embedding \
    --device cuda:3 \
    --load /data/rrjin/NMT/data/bible_data/models/attention_model/attention_multi_gpu_lstm__8_2.988244 \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/attention_model/attention_src_combine_32000.vocab \
    --lang_token_list_path /data/rrjin/NMT/data/bible_data/bible_lang_tok \
    --lang_vec_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/attention_model/bible_lang_vec_attention_rnn.token_embedding