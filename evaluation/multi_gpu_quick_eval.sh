cd ..

model_load=/data/rrjin/NMT/data/ted_data/models/transformer/transformer_6_17563
test_src_path=/data/rrjin/NMT/data/ted_data/corpus/test_data_src_joint_bpe_32000.combine
src_vocab_path=/data/rrjin/NMT/data/ted_data/vocab_data/transformer/src_32000_transformer.combine.vocab
tgt_vocab_path=/data/rrjin/NMT/data/ted_data/vocab_data/transformer/tgt_32000_transformer.en.vocab
translation_output_dir=/data/rrjin/NMT/data/ted_data/translation/transformer_test_beam_size_5

python -u -m evaluation.multi_gpu_quick_eval \
    --device_id 0 1 2 3 \
    --model_load ${model_load} \
    --transformer \
    --test_src_path ${test_src_path} \
    --src_vocab_path ${src_vocab_path} \
    --tgt_vocab_path ${tgt_vocab_path} \
    --translation_output_dir ${translation_output_dir} \
    --record_time \
    --need_tok \
    --beam_size 5
