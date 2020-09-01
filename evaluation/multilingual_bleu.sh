cd ..

python -u -m evaluation.multilingual_bleu \
    --src_file_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_test_data_src_tok_joint_bpe_32000_filtered1.combine \
    --translation_path_list \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_test_beam_size_5_merged/transformer_6_17861_beam_size5_translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_test_data_tgt_tok_filtered1.en \
    --language_data /data/rrjin/NMT/data/ted_data_new/ted_lang_tok_to_languages.json \
    --bleu_score_data_path /data/rrjin/NMT/data/ted_data_new/translation/transformer_test_beam_search_5_bleu_score_statistic.json