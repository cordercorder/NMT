cd ..

python -u -m data_processing.convert_iso639code_to_language \
    --language_code_path /data/rrjin/NMT/data/ted_data_new/ted_lang_tok \
    --output_file_path /data/rrjin/NMT/data/ted_data_new/ted_lang_tok_to_languages.json