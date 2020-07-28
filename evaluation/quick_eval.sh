cd ..

python -u -m evaluation.quick_eval \
    --device cuda:3 \
    --model_prefix /data/rrjin/NMT/data/bible_data/models/attention_model/attention_multi_gpu_lstm__8_2.988244 \
    --test_src_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/test_src_combine_bpe_32000_remove_at.txt \
    --test_tgt_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/test_tgt_en_bpe_32000_remove_at.txt \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/attention_model/attention_src_combine_32000.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/attention_model/attention_tgt_en_32000.vocab \
    --translation_output /data/rrjin/NMT/data/bible_data/translation/attention_model_test