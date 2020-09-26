cd ..

python -u -m evaluation.multilingual_bleu \
    --src_file_path /data/rrjin/NMT/data/ted_data/corpus/test_data_src_joint_bpe_32000.combine \
    --translation_path_list \
      /data/rrjin/NMT/data/ted_data/translation/transformer_label_smoothing_test_beam_size_4_merged/transformer_12_17563_beam_size4_translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data/corpus/test_data_tgt_joint_bpe_32000.en \
    --language_data /data/rrjin/NMT/data/ted_data/ted_lang_tok_to_languages.json \
    --bleu_score_data_path /data/rrjin/NMT/data/ted_data/translation/transformer_label_smoothing_test_beam_search_4_bleu_score_statistic.json \
    --multilingual \
    --bleu_score_type sacrebleu
