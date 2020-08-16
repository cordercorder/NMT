cd ..

python -m lang_vec.langvec_classify \
    --feature_name syntax_wals \
    --classify_method svm \
    --lang_vec_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/transformer/bible_lang_vec_transformer.token_embedding \
    --lang_name_path /data/rrjin/NMT/data/bible_data/bible_lang_tok \
    --output_file_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/transformer/langvec_svm_classify_syntax_wals.txt