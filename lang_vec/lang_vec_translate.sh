cd ..

python -u -m lang_vec.lang_vec_translate \
    --device cuda:3 \
    --load /data/rrjin/NMT/data/bible_data/models/transformer/bible_transformer_single_gpu__9_0.138838 \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/transformer/src_combine_32000_transformer.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/transformer/tgt_en_32000_transformer.vocab \
    --test_src_path /data/rrjin/NMT/data/bible_data/corpus/test_src_combine_joint_bpe_22000.txt \
    --test_tgt_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/test_tgt_en.txt \
    --lang_vec_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/transformer/bible_lang_vec_transformer.token_embedding \
    --translation_output /data/rrjin/NMT/data/bible_data/translation/transformer/bible_lang_vec_transformer_token_embedding_test_translation \
    --transformer