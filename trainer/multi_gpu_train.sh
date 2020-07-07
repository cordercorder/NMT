python -m torch.distributed.launch --nproc_per_node=3 multi_gpu_train.py \
    --device_id 1 2 3 \
    --src_language spa \
    --tgt_language en \
    --src_path /data/rrjin/NMT/tmpdata/train_src.spa \
    --tgt_path /data/rrjin/NMT/tmpdata/train_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --rnn_type lstm \
    --embedding_size 512 \
    --hidden_size 512 \
    --num_layers 3 \
    --checkpoint /data/rrjin/NMT/tmpdata/model_basic_multi_gpu_lstm6 \
    --batch_size 64 \
    --dropout 0.1 \
    --epoch 1 \
    --load /data/rrjin/NMT/tmpdata/model_basic_multi_gpu_lstm6__0_2.100440