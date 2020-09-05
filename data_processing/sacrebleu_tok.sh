cd ..

corpus=/data/rrjin/corpus_data/ted_data
tokenized_corpus_dir_prefix=/data/rrjin/corpus_data/tokenized_ted_data

for directory in `ls ${corpus}`; do

    raw_corpus_dir=${corpus}/${directory}
    tokenized_corpus_dir=${tokenized_corpus_dir_prefix}/${directory}

    echo ${raw_corpus_dir}

    python -u -m data_processing.sacrebleu_tok \
        --raw_corpus_dir ${raw_corpus_dir} \
        --tokenized_corpus_dir ${tokenized_corpus_dir}

done
