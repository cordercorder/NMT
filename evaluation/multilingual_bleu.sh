cd ..

python -u -m evaluation.multilingual_bleu \
    --lang_identifier_file_path /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/data/dev_data_max_len_150_sorted_lang_code.combine \
    --translation_path_list \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_0_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_10_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_11_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_12_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_13_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_1_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_14_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_2_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_3_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_4_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_5_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_6_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_7_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_8_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_merged_remove_lang_code/transformer_9_4057_greedy_translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/data/dev_data_max_len_150_sorted_remove_lang_code.combine \
    --language_data /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/ted_lang_tok_to_languages.json \
    --bleu_score_data_path /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/translation/transformer_label_smoothing_dev_greedy_bleu_score_statistic.json \
    --multilingual \
    --bleu_score_type sacrebleu
