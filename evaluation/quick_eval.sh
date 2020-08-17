cd ..

python -u -m evaluation.quick_eval \
    --device cuda:1 \
    --model_load /data/rrjin/NMT/data/bible_data/models/basic_model/basic_multi_gpu_lstm_9_41036_old \
    --test_src_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/test_src_combine_bpe_32000_remove_at.txt \
    --test_tgt_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/test_tgt_en_bpe_32000_remove_at.txt \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/basic_model/basic_src_combine_32000_old.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/basic_model/basic_tgt_en_32000_old.vocab \
    --translation_output_dir /data/rrjin/NMT/data/bible_data/translation/basic_model_test \
    --record_time