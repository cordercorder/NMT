#nohup bash quick_eval.sh > /data/rrjin/NMT/data/ted_data/evaluation_logs/transformer/eval_transformer_ted_test_data_logs_beam_size_5.txt 2>&1 &
#nohup bash multi_gpu_quick_eval.sh > /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/evaluation_logs/transformer_label_smoothing/eval_transformer_label_smoothing_sub_ted_dev_data_logs_greedy.txt 2>&1 &
nohup bash multilingual_bleu.sh > /data/rrjin/NMT/data/ted_data/sub_corpus_both_with_lang_code/evaluation_logs/transformer_label_smoothing/dev_greedy_bleu_statistic.txt 2>&1 &
