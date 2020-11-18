cd ..

python -m trainer.train_transformer \
    --device cuda:3 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --src_language fa \
    --tgt_language zh \
    --src_path /data/cordercorder/NMT/data/corpus/clean.tok.pe_max_len_80.32000 \
    --tgt_path /data/cordercorder/NMT/data/corpus/clean.tok.zh_max_len_80.32000 \
    --src_vocab_path /data/cordercorder/NMT/data/corpus/pe.32000.vocab \
    --tgt_vocab_path /data/cordercorder/NMT/data/corpus/zh.32000.vocab \
    --checkpoint /data/cordercorder/NMT/data/model_checkpoint/transformer \
    --dropout 0.1 \
    --learning_rate 0.0005 \
    --optim_method adam_inverse_sqrt \
    --end_epoch 20 \
    --batch_size 64 \
    --rebuild_vocab \
    --update_freq 4
