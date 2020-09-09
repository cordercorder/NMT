cd ..

python -u -m evaluation.multilingual_bleu \
    --src_file_path /data/rrjin/NMT/data/ted_data/corpus/dev_data_src_joint_bpe_32000_max_len_150_sorted.combine \
    --translation_path_list \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_19_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_18_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_17_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_16_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_15_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_14_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_13_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_12_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_11_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_10_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_9_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_8_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_7_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_6_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_5_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_4_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_3_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_2_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_1_17563_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_merged/transformer_0_17563_translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data/corpus/dev_data_tgt_max_len_150_sorted.en \
    --language_data /data/rrjin/NMT/data/ted_data/ted_lang_tok_to_languages.json \
    --bleu_score_data_path /data/rrjin/NMT/data/ted_data/translation/transformer_dev_greedy_bleu_score_statistic.json \
    --multilingual \
    --bleu_score_type sacrebleu
