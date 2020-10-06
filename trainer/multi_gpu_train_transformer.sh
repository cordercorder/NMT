cd ..

python -m trainer.multi_gpu_train_transformer \
    --device_id 0 1 2 3 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --src_language en \
    --tgt_language combine \
    --src_path /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/data/train_data_joint_bpe_32000_max_len_100_shuffled.en \
    --tgt_path /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/data/train_data_joint_bpe_32000_max_len_100_shuffled.combine \
    --src_vocab_path /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/vocab_data/transformer_label_smoothing/src_32000_transformer.en.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/vocab_data/transformer_label_smoothing/tgt_32000_transformer.combine.vocab \
    --checkpoint /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/models/transformer_label_smoothing/transformer \
    --dropout 0.1 \
    --learning_rate 0.00005 \
    --end_epoch 15 \
    --batch_size 64 \
    --rebuild_vocab
