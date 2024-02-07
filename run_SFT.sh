nohup accelerate launch --num_processes 4 --gpu_ids 1,0,2,5 --main_process_port 2026 train.py \
--seed 0 \
--data_path data/dataset/steam/ \
--output snap/0206-steam/ \
--backbone /home/wangshuo/weights/llama2/Llama-2-7b-hf-chat/ \
--item_index title \
--batch_size 1 \
--topk 10 \
--clip_grad_norm 1.0 \
--epoch 40 \
--gen_max_length 512 \
--lr 0.0006 \
--gradient_accumulation_steps 12 \
--train_stage SFT \
--SFT_actor_lora_r 16 \
--warmup_ratio 0.0125 \
--val_batch_size 12 \
--SFT_train_tasks SFTSeqRec,SFTControlRec \
--SFT_val_tasks SFTTestSeqRec \
--backup_ip 0.0.0.0 \
--val_epoch 5 \
--share_chat_gpt_ratio 0.5 \
--llama2_chat_template \
--user_control_symbol \
--idx>snap/0206-steam/output.log 2>&1 &

# --SFT_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch07_SFT 
# --FA2 \