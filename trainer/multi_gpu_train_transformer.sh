cd ..

python -m trainer.multi_gpu_train_transformer \
    --device_id 0 1 2 3 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --src_language combine \
    --tgt_language en \
    --src_path /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_src_tok_joint_bpe_32000_filtered.combine \
    --tgt_path /data/rrjin/NMT/data/ted_data_new/corpus/shuffled_train_data_tgt_tok_joint_bpe_32000_filtered.en \
    --src_vocab_path /data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/src_32000_transformer.combine.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/tgt_32000_transformer.en.vocab \
    --checkpoint /data/rrjin/NMT/data/ted_data_new/models/transformer \
    --dropout 0.1 \
    --rebuild_vocab \
    --learning_rate 0.00005 \
    --end_epoch 20 \
    --batch_size 64