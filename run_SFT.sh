nohup accelerate launch --num_processes 4 --gpu_ids 1,3,4,5 --main_process_port 2036 train.py \
--seed 0 \
--data_path data/dataset/toys/ \
--output snap/0414-toys-method3-ctrl/ \
--backbone /home/wangshuo/weights/llama2/Llama-2-7b-hf-chat/ \
--item_index title \
--batch_size 1 \
--topk 10 \
--clip_grad_norm 1.0 \
--epoch 40 \
--gen_max_length 512 \
--max_token_length 512 \
--lr 0.0006 \
--gradient_accumulation_steps 16 \
--train_stage SFT \
--SFT_actor_lora_r 16 \
--warmup_ratio 0.0125 \
--val_batch_size 12 \
--SFT_train_tasks SFTSeqRec \
--SFT_val_tasks SFTTestSeqRec \
--backup_ip 0.0.0.0 \
--val_epoch 5 \
--share_chat_gpt_ratio 0.0 \
--llama2_chat_template \
--lm_head \
--idx>>snap/0414-toys-method3-ctrl/output.log 2>&1 &

# --SFT_load /home/wangshuo/codes/scope-rec/snap/0411-toys-method1/Epoch06_SFT
# --FA2 \
#--user_control_symbol \
#--use_scope_mask \
#--user_control_symbol \
# ssh -L 2025:localhost:2025 wangshuo@172.31.226.127 -P 50805