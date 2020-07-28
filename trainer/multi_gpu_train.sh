cd ..

python -m trainer.multi_gpu_train \
    --device_id 2 3 \
    --src_language combine \
    --tgt_language en \
    --src_path /data/rrjin/NMT/data/bible_data/corpus/train_src_combine_joint_bpe_22000_filtered.txt \
    --tgt_path /data/rrjin/NMT/data/bible_data/corpus/train_tgt_en_joint_bpe_22000_filtered.txt \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/attention_model/attention_src_combine_32000.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/attention_model/attention_tgt_en_32000.vocab \
    --rnn_type lstm \
    --embedding_size 512 \
    --hidden_size 512 \
    --num_layers 3 \
    --checkpoint /data/rrjin/NMT/data/bible_data/models/attention_model/attention_multi_gpu_lstm \
    --batch_size 64 \
    --dropout 0.1 \
    --rebuild_vocab \
    --learning_rate 0.00005 \
    --attention_size 512