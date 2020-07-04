python -m evaluation.eval \
    --device cuda:3 \
    --load /data/rrjin/NMT/tmpdata/model_basic_lstm__9_1.631652 \
    --test_src_path /data/rrjin/NMT/tmpdata/val_src.spa \
    --test_tgt_path /data/rrjin/NMT/tmpdata/val_tgt.en \
    --src_vocab_path /data/rrjin/NMT/tmpdata/src.spa.vocab \
    --tgt_vocab_path /data/rrjin/NMT/tmpdata/tgt.en.vocab \
    --translation_output /data/rrjin/NMT/tmpdata/tmp_beam_translations_basic_lstm2.txt