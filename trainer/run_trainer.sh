# nohup bash train_transformer.sh > ../logs/test_spa_en_transformer_single_gpu_train_logs_0.00005.txt 2>&1 &

# nohup bash multi_gpu_train_transformer.sh > ../logs/test_transformer_multi_gpu_barrier_train_logs_0.00005.txt 2>&1 &

nohup bash multi_gpu_train.sh> ../logs/basic_model/basic_multi_gpu_lstm_lstm_0.0005.txt 2>&1 &