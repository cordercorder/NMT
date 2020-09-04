cd ..

lang_vec_dir=/data/rrjin/NMT/data/ted_data_new/lang_vec/transformer
picture_dir=/data/rrjin/NMT/data/ted_data_new/lang_vec_picture/transformer

for lang_vec_file in `ls ${lang_vec_dir}`; do

    lang_vec_path=${lang_vec_dir}/${lang_vec_file}
    picture_path=${picture_dir}/${lang_vec_file}.jpg

    python -u -m lang_vec.lang_vec_plot \
        --lang_vec_path ${lang_vec_path} \
        --picture_path ${picture_path}
done
