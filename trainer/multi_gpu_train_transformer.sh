cd ..

python -m trainer.multi_gpu_train_transformer \
    --device_id 0 1 2 3 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --src_language combine \
    --tgt_language en \
    --src_path /data/rrjin/NMT/data/ted_data/corpus/train_data_src_joint_bpe_32000_max_len_100_shuffled.combine \
    --tgt_path /data/rrjin/NMT/data/ted_data/corpus/train_data_tgt_joint_bpe_32000_max_len_100_shuffled.en \
    --src_vocab_path /data/rrjin/NMT/data/ted_data/vocab_data/transformer/src_32000_transformer.combine.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/ted_data/vocab_data/transformer/tgt_32000_transformer.en.vocab \
    --checkpoint /data/rrjin/NMT/data/ted_data/models/transformer_label_smoothing/transformer \
    --dropout 0.1 \
    --learning_rate 0.00005 \
    --end_epoch 20 \
    --batch_size 64
