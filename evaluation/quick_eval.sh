cd ..

python -u -m evaluation.quick_eval \
    --device cuda:1 \
    --model_load /data/rrjin/NMT/data/ted_data/models/transformer/transformer_rank0 \
    --transformer \
    --test_src_path /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_src_tok_joint_bpe_32000.combine \
    --test_tgt_path /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_tgt_tok.en \
    --src_vocab_path /data/rrjin/NMT/data/ted_data/vocab_data/transformer/src_32000_transformer.combine.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/ted_data/vocab_data/transformer/tgt_32000_transformer.en.vocab \
    --translation_output_dir /data/rrjin/NMT/data/ted_data/translation/transformer_test \
    --record_time \
    --need_tok \
    --beam_size 5