cd ..

model_load=/data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/models/transformer_label_smoothing/transformer
test_src_path=/data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/data/dev_data_joint_bpe_32000_max_len_150_sorted.en
src_vocab_path=/data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/vocab_data/transformer_label_smoothing/src_32000_transformer.en.vocab
tgt_vocab_path=/data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/vocab_data/transformer_label_smoothing/tgt_32000_transformer.combine.vocab
translation_output_dir=/data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy


python -u -m evaluation.multi_gpu_quick_eval \
    --device_id 0 1 2 3 \
    --model_load ${model_load} \
    --is_prefix \
    --transformer \
    --test_src_path ${test_src_path} \
    --src_vocab_path ${src_vocab_path} \
    --tgt_vocab_path ${tgt_vocab_path} \
    --translation_output_dir ${translation_output_dir} \
    --record_time \
    --batch_size 32 \
    --need_tok \
    --work_load_per_process 30 10 5 1
