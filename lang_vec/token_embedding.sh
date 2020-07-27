cd ..

python -m lang_vec.token_embedding \
    --device cuda:3 \
    --load /data/rrjin/NMT/data/bible_data/models/transformer/bible_transformer_single_gpu__9_0.138838 \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/transformer/src_combine_32000_transformer.vocab \
    --lang_token_list_path /data/rrjin/NMT/data/bible_data/bible_lang_tok \
    --lang_vec_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/transformer/bible_lang_vec_transformer.token_embedding \
    --transformer \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/transformer/tgt_en_32000_transformer.vocab