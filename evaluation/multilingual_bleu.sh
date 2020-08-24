cd ..

python -u -m evaluation.multilingual_bleu \
    --src_file_path /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_src_tok_joint_bpe_32000_sorted_filter2.combine \
    --translation_path_list /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer_0_21442_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer__0_3.454432_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer_1_21442_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer__1_2.214248_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer_10_21442_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer__10_0.787453_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer_11_21442_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer__11_0.750704_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer_12_21442_translations_tok.txt \
                            /data/rrjin/NMT/data/ted_data/translation/transformer_dev_batch_translation_sorted/transformer__12_0.719120_translations_tok.txt \
    --reference_path /data/rrjin/NMT/data/ted_data/corpus/raw_dev_data_tgt_tok_sorted_filter2.en \
    --language_data /data/rrjin/NMT/data/ted_data/ted_lang_tok_to_languages.json