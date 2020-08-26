cd ..

python -u -m evaluation.multi_gpu_quick_eval \
    --device_id 0 1 2 3 \
    --model_load /data/rrjin/NMT/data/ted_data_new/models/transformer/transformer \
    --transformer \
    --is_prefix \
    --test_src_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_src_tok_joint_bpe_32000_filtered2_sorted.combine \
    --test_tgt_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_tgt_tok_filtered2_sorted.en \
    --src_vocab_path /data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/src_32000_transformer.combine.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/tgt_32000_transformer.en.vocab \
    --translation_output_dir /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_greedy \
    --record_time \
    --need_tok \
    --batch_size 64