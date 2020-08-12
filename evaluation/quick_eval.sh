cd ..

python -u -m evaluation.quick_eval \
    --device cuda:2 \
    --model_load /data/rrjin/NMT/tmpdata/model_attention_lstm__9_0.272557 \
    --test_src_path /data/rrjin/NMT/tmpdata/test_src_tok.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/test_tgt_tok.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src2.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt2.en.vocab \
    --translation_output_dir /data/rrjin/NMT/tmpdata/temp_model_attention \
    --beam_size 3 \
    --record_time