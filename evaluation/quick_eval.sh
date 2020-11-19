cd ..

python -u -m evaluation.quick_eval \
    --device cuda:3 \
    --model_load /data/cordercorder/NMT/data/model_checkpoint/transformer \
    --is_prefix \
    --transformer \
    --test_src_path /data/cordercorder/NMT/data/corpus/test.pe.32000 \
    --test_tgt_path /data/cordercorder/NMT/data/corpus/test.tok.zh \
    --src_vocab_path /data/cordercorder/NMT/data/corpus/pe.32000.vocab \
    --tgt_vocab_path /data/cordercorder/NMT/data/corpus/zh.32000.vocab \
    --translation_output_dir /data/cordercorder/NMT/data/model_translation/transformer_base \
    --record_time \
    --need_tok \
    --beam_size 5 \
    --bleu_script_path /data/cordercorder/NMT/scripts/multi-bleu.perl
