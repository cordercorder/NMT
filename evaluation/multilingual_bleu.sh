cd ..

python -u -m evaluation.multilingual_bleu \
    --src_file_path /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_src_tok_joint_bpe_32000.combine \
    --translation_path_list /data/rrjin/NMT/data/ted_data/translation/transformer_rank0_beam_size5__translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data/corpus/raw_test_data_tgt_tok.en \
    --language_data /data/rrjin/NMT/data/ted_data/ted_lang_tok_to_languages.json