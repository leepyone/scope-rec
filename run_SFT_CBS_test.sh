nohup python train.py \
--seed 0 \
--data_path /home/wangshuo/codes/scope-rec/data/dataset/movies/ \
--backbone /home/wangshuo/weights/llama2/Llama-2-7b-hf-chat/ \
--item_index title \
--test_batch_size 16 \
--topk 10 \
--gpu cuda:0 \
--gen_max_length 512 \
--train_stage SFT_Test \
--SFT_actor_lora_r 16 \
--SFT_test_task SFTTestSeqRec \
--backup_ip 0.0.0.0 \
--idx \
--use_CBS \
--user_control_symbol \
--SFT_load /home/wangshuo/codes/scope-rec/snap/0322-movies-trl/Epoch40_SFT>/home/wangshuo/codes/scope-rec/snap/0322-movies-trl/Epoch40_SFT_CBS_test_output.log 2>&1 &
# --FA2
# --use_CBS \
#--user_control_symbol \