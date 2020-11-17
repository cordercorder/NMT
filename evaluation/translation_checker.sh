cd ..

python -u -m evaluation.translation_checker \
    --translation_file_path_list \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_0_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_1_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_2_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_3_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_4_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_5_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_6_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_7_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_8_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_9_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_10_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_11_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_12_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_13_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_un_tok_merged_remove_lang_code/transformer_14_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_0_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_1_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_2_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_3_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_4_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_5_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_6_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_7_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_8_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_9_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_10_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_11_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_12_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_13_4057_greedy_translations.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_lang_code_greedy_un_tok_merged_remove_lang_code/transformer_14_4057_greedy_translations.txt \
    --lang_identifier_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data_max_len_150_sorted_lang_code.combine \
    --operation check_word \
    --vocab_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/train_data_joint_bpe_32000_max_len_100_shuffled.combine.vocab
