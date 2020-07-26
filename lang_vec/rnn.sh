cd ..

python -u -m lang_vec.rnn \
    --device cuda:3 \
    --load /data/rrjin/NMT/data/bible_data/models/attention_model/attention_multi_gpu_lstm__8_2.988244 \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/attention_model/attention_src_combine_32000.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/attention_model/attention_tgt_en_32000.vocab \
    --test_src_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/test_src_combine_bpe_32000_remove_at.txt \
    --test_tgt_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/test_tgt_en_bpe_32000_remove_at.txt \
    --lang_vec_path /data/rrjin/NMT/data/bible_data/lang_vec_token_embedding/attention_model/bible_lang_vec_attention_rnn.token_embedding \
    --translation_output /data/rrjin/NMT/data/bible_data/translation/attention_model/bible_lang_vec_attention_rnn_token_embedding_test_translation