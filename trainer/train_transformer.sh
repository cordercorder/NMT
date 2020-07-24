cd ..

python -m trainer.train_transformer \
    --device cuda:2 \
    --num_layers 6 \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --src_language combine \
    --tgt_language en \
    --src_path /data/rrjin/NMT/data/bible_data/corpus/train_src_combine_joint_bpe_22000_filtered.txt \
    --tgt_path /data/rrjin/NMT/data/bible_data/corpus/train_tgt_en_joint_bpe_22000_filtered.txt \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/test_src_combine_32000_transformer.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/test_tgt_en_32000_transformer.vocab \
    --checkpoint /data/rrjin/NMT/data/bible_data/models/test_bible_transformer_single_gpu \
    --dropout 0.1 \
    --rebuild_vocab \
    --learning_rate 0.00005 \
    --batch_size 64