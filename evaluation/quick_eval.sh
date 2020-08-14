cd ..

python -u -m evaluation.quick_eval \
    --device cuda:2 \
    --model_load /data/rrjin/NMT/data/bible_data/models/basic_model/basic_multi_gpu_lstm_rank0 \
    --test_src_path /data/rrjin/NMT/data/bible_data/corpus/test_src_combine_joint_bpe_22000.txt \
    --test_tgt_path /data/rrjin/NMT/data/bible_data/corpus/test_tgt_en_joint_bpe_22000.txt \
    --src_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/basic_model/basic_src_combine_32000.vocab \
    --tgt_vocab_path /data/rrjin/NMT/data/bible_data/vocab_data/basic_model/basic_tgt_en_32000.vocab \
    --translation_output_dir /data/rrjin/NMT/tmpdata/temp_model_basic_old \
    --beam_size 3 \
    --record_time