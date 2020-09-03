cd ..

python -u -m evaluation.multilingual_bleu \
    --src_file_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_src_tok_joint_bpe_32000_filtered2.combine \
    --translation_path_list \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_2_merged/transformer_6_17861_beam_size2_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_3_merged/transformer_6_17861_beam_size3_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_4_merged/transformer_6_17861_beam_size4_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_5_merged/transformer_6_17861_beam_size5_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_6_merged/transformer_6_17861_beam_size6_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_7_merged/transformer_6_17861_beam_size7_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_8_merged/transformer_6_17861_beam_size8_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_9_merged/transformer_6_17861_beam_size9_translations_tok.txt \
      /data/rrjin/NMT/data/ted_data_new/translation/transformer_dev_beam_size_10_merged/transformer_6_17861_beam_size10_translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data_new/corpus/raw_dev_data_tgt_tok_filtered2.en \
    --language_data /data/rrjin/NMT/data/ted_data_new/ted_lang_tok_to_languages.json \
    --bleu_score_data_path /data/rrjin/NMT/data/ted_data_new/translation/transformer_test_beam_search_2_to_10_bleu_score_statistic.json \
    --multilingual