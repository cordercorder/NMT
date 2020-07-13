python -u -m evaluation.quick_eval \
    --device cuda:0 \
    --model_prefix /data/rrjin/NMT/data/models/basic_multi_gpu_lstm \
    --test_src_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/dev_src_combine_bpe_32000.txt \
    --test_tgt_path /data/rrjin/corpus_data/lang_vec_data/bible-corpus/train_data/dev_tgt_en_bpe_32000.txt \
    --src_vocab_path /data/rrjin/NMT/data/src_combine_32000.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/tgt_en_32000.vocab \
    --translation_output /data/rrjin/NMT/data/translation