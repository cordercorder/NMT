cd ..

python -u -m evaluation.multilingual_bleu \
    --src_file_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data_max_len_150_sorted.combine \
    --translation_path_list \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_0_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_10_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_11_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_12_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_13_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_1_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_14_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_2_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_3_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_4_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_5_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_6_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_7_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_8_4057_greedy_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_merged/transformer_9_4057_greedy_translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data/sub_corpus/data/dev_data_max_len_150_sorted.combine \
    --language_data /data/rrjin/NMT/data/ted_data/sub_corpus/ted_lang_tok_to_languages.json \
    --bleu_score_data_path /data/rrjin/NMT/data/ted_data/sub_corpus/translation/transformer_label_smoothing_dev_greedy_bleu_score_statistic.json \
    --multilingual \
    --bleu_score_type sacrebleu
