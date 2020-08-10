cd ..

python -m lang_vec.langvec_logistic_classify \
    --feature_name syntax_wals \
    --lang_vec_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/attention_model/bible_lang_vec_attention_rnn.token_embedding \
    --lang_name_path /data/rrjin/NMT/data/bible_data/bible_lang_tok \
    --output_file_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/attention_model/langvec_logistic_classify_syntax_wals.txt