cd ..

python -m lang_vec.rnn_basic \
    --device cuda:2 \
    --load /data/rrjin/NMT/data/models/basic_multi_gpu_lstm__9_3.178944 \
    --src_vocab_path /data/rrjin/NMT/data/src_combine_32000.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/tgt_en_32000.vocab \
    --test_src_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/dev_src_combine_bpe_32000.txt \
    --test_tgt_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/dev_tgt_en_bpe_32000.txt \
    --lang_vec_path /data/rrjin/NMT/data/bible_lang_vec_basic_rnn.token_embedding \
    --translation_output /data/rrjin/NMT/data/translation/bible_lang_vec_basic_rnn_token_embedding_dev_translation