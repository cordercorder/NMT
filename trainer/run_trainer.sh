# nohup bash train_transformer.sh > ../logs/test_spa_en_transformer_single_gpu_train_logs_0.00005.txt 2>&1 &

nohup bash multi_gpu_train_transformer.sh > /data/rrjin/NMT/data/ted_data/sub_corpus/train_logs/transformer_label_smoothing/transformer_multi_gpu_train_sub_ted_logs_0.00005_label_smoothing.tok13a.txt 2>&1 &

# nohup bash multi_gpu_train.sh> ../logs/basic_model/basic_multi_gpu_lstm_lstm_0.0005.txt 2>&1 &