cd ..

model_load=/data/rrjin/NMT/data/ted_data_new/models/transformer/transformer_6_17861
test_src_path=/data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_src_tok_joint_bpe_32000_filtered2.combine
test_tgt_path=/data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_tgt_tok_filtered2.en
src_vocab_path=/data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/src_32000_transformer.combine.vocab
tgt_vocab_path=/data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/tgt_32000_transformer.en.vocab
translation_output_dir_prefix=/data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size

for i in 2 9 10; do

  python -u -m evaluation.multi_gpu_quick_eval \
      --device_id 0 1 2 3 \
      --model_load ${model_load} \
      --transformer \
      --test_src_path ${test_src_path} \
      --test_tgt_path ${test_tgt_path} \
      --src_vocab_path ${src_vocab_path} \
      --tgt_vocab_path ${tgt_vocab_path} \
      --translation_output_dir ${translation_output_dir_prefix}_${i} \
      --record_time \
      --need_tok \
      --beam_size ${i}
done
