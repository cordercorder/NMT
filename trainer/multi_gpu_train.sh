cd ..

python -m trainer.multi_gpu_train \
    --device_id 1 2 3 \
    --src_language spa \
    --tgt_language en \
    --src_path /data/rrjin/NMT/tmpdata/train_src_tok.spa \
    --tgt_path /data/rrjin/NMT/tmpdata/train_tgt_tok.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/test_src.spa2.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/test_tgt.en2.vocab \
    --rnn_type lstm \
    --embedding_size 512 \
    --hidden_size 512 \
    --num_layers 3 \
    --checkpoint /data/rrjin/NMT/tmpdata/test_new_attention_multi_gpu_lstm2 \
    --batch_size 32 \
    --dropout 0.2 \
    --rebuild_vocab \
    --attention_size 512