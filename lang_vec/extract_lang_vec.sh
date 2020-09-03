cd ..

load=/data/rrjin/NMT/data/ted_data_new/models/transformer/transformer_6_17861
src_vocab_path=/data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/src_32000_transformer.combine.vocab
lang_token_list_path=/data/rrjin/NMT/data/ted_data_new/ted_lang_tok
lang_vec_path_prefix=/data/rrjin/NMT/data/ted_data_new/lang_vec/transformer/lang_vec
lang_vec_type=layer_token_embedding
tgt_vocab_path=/data/rrjin/NMT/data/ted_data_new/vocab_data/transformer/tgt_32000_transformer.en.vocab

for layer_id in {0..5}; do
    python -m lang_vec.extract_lang_vec \
        --load ${load} \
        --src_vocab_path ${src_vocab_path} \
        --lang_token_path ${lang_token_list_path} \
        --lang_vec_path ${lang_vec_path_prefix}_${layer_id}.layer_token_embedding \
        --transformer \
        --lang_vec_type ${lang_vec_type} \
        --tgt_vocab_path ${tgt_vocab_path} \
        --layer_id ${layer_id}
done
