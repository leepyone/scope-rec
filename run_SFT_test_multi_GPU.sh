nohup accelerate launch --num_processes 2 --gpu_ids 3,4 --main_process_port 2027 train.py \
--seed 0 \
--data_path /home/wangshuo/codes/InstructControllableRec_RLHF/data/dataset/movies/ \
--output /home/wangshuo/codes/InstructControllableRec_RLHF/snap/0207-steam-single/test-epoch32/CBS_test/ \
--backbone /home/wangshuo/weights/llama2/Llama-2-7b-hf-chat/ \
--item_index title \
--test_batch_size 14 \
--topk 10 \
--gpu cuda:0 \
--gen_max_length 300 \
--train_stage SFT_Test \
--SFT_actor_lora_r 16 \
--SFT_test_task SFTTestSeqRec \
--backup_ip 0.0.0.0 \
--user_control_symbol \
--idx \
--use_CBS \
--SFT_load /home/wangshuo/codes/InstructControllableRec_RLHF/snap/0212-movies-ctrl-c/Epoch40_SFT> /home/wangshuo/codes/InstructControllableRec_RLHF/snap/0212-movies-ctrl-c/epoch40_CBS_test_output.log 2>&1 &
# --FA2
# --use_CBS \