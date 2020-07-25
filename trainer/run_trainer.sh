# nohup bash train_transformer.sh > ../logs/test_combine_transformer_single_gpu_train_logs_0.00005.txt 2>&1 &

nohup bash multi_gpu_train_transformer.sh > ../logs/test_transformer_multi_gpu_barrier_train_logs_0.00005.txt 2>&1 &

# nohup bash multi_gpu_train.sh> ../logs/test_attention_new_multi_gpu_train_logs2.txt 2>&1 &