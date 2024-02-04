# SFT stage
### SFT Train
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_title64_t_0_q_llama7b_CR/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--batch_size 1 
--topk 10 
--gpu cuda:0 
--clip_grad_norm 1.0 
--epoch 10 
--gen_max_length 512 
--quantization 
--lr 0.0005 
--gradient_accumulation_steps 16 
--train_stage SFT 
--SFT_actor_lora_r 16 
--log_to_file
--SFT_train_tasks SFTSeqRec,SFTControlRec,SFTPersonalControlRec,SFTCategoryRate
--SFT_val_tasks SFTValSeqRec,SFTValControlRec,SFTValPersonalControlRec
--FA2
```

### SFT Train Continue
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_title64_t_0_q_llama7b_CR/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--batch_size 1 
--topk 10 
--gpu cuda:0 
--clip_grad_norm 1.0 
--epoch 10 
--gen_max_length 512 
--quantization 
--lr 0.0005 
--warmup_ratio 0.025
--gradient_accumulation_steps 16 
--train_stage SFT 
--SFT_actor_lora_r 16 
--log_to_file
--SFT_load {xxx}
--SFT_train_tasks SFTSeqRec,SFTControlRec,SFTPersonalControlRec,SFTCategoryRate
--SFT_val_tasks SFTValSeqRec,SFTValControlRec,SFTValPersonalControlRec
--FA2
```

### SFT Test after Merge
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Q_Llama7b_SC/ 
--backbone {xxx}
--item_index title64_t 
--test_batch_size 16 
--topk 10 
--gpu cuda:0 
--gen_max_length 512 
--quantization 
--train_stage SFT_Test
--SFT_actor_lora_r 0 
--SFT_test_task SFTTestSeqRec 
--backup_ip 0.0.0.0 
--FA2
```

### SFT Test before Merge
```shell
python train.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Q_Llama7b_SC/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--test_batch_size 16 
--topk 10 
--gpu cuda:0 
--gen_max_length 512 
--quantization 
--train_stage SFT_Test 
--SFT_actor_lora_r 16 
--SFT_test_task SFTTestSeqRec 
--backup_ip 0.0.0.0
--SFT_load {xxx} 
--FA2
```


# SFT model Merge
```shell
python train.py 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Q_Llama7b/ 
--backbone meta-llama/Llama-2-7b-hf 
--item_index title64_t 
--gpu cuda:0 
--train_stage SFT_Merge 
--SFT_actor_lora_r 16 
--SFT_load snap/ICR_SubMovie_Title64T_0_Q_Llama7b/Epoch20_SFT
```


# RLHF Stage
### RLHF Train
```shell
python train.py 
--seed 0
--data_path data/dataset/sub_movie/
--output snap/ICR_SubMovie_title64_t_0_q_llama7b_CR1/
--backbone snap/ICR_SubMovie_title64_t_0_q_llama7b_CR1/SFT_Epoch40/
--item_index title64_t
--batch_size 8
--sample_num 2
--gradient_accumulation_steps 4
--topk 10
--gpu cuda:0
--clip_grad_norm 0.5
--epoch 4
--gen_max_length 512
--train_stage RLHF
--RLHF_actor_lora_r 4
--RLHF_critic_lora_r 4
--RLHF_train_tasks RLHFSeqRec,RLHF+PersonalControlRec,RLHF-PersonalControlRec
--RLHF_val_tasks RLHFSeqRec,RLHF+PersonalControlRec,RLHF-PersonalControlRec
--backup_ip 0.0.0.0
--lr 0.00005
--lora_dropout 0.1
--weight_decay 0.01
--kl_coef 0.3
--entropy_weight 0.01
--vf_coef 0.1
--policy_kl_threshold 0.03
--quantization
--lm_head
--fine_grain_reward
```

### RLHF Test 
```shell
python train.py 
--seed 0
--data_path data/dataset/sub_movie/
--output snap/ICR_SubMovie_title64_t_0_q_llama7b_CR1/
--backbone snap/ICR_SubMovie_title64_t_0_q_llama7b_CR1/SFT_Epoch40/
--item_index title64_t
--topk 10
--gpu cuda:0
--gen_max_length 512
--train_stage RLHF_Test
--RLHF_actor_lora_r 4
--RLHF_critic_lora_r 4
--RLHF_test_task RLHFSeqRec
--backup_ip 0.0.0.0
--quantization
--lm_head
--fine_grain_reward
--RLHF_load {xxx}
```


# MMLU test
### MMLU test of Llama-2-7b model
```shell
python MMLU.py
--seed 0 
--backbone snap/Llama-2-7b-hf 
--gpu cuda:1 
--max_token_length 2048 
--quantization 
--train_stage SFT_Test 
--SFT_actor_lora_r 0 
--FA2
```
### MMLU test of SFT model
```shell
python MMLU.py
--seed 0 
--backbone snap/Llama-2-7b-hf 
--gpu cuda:1 
--max_token_length 2048 
--quantization 
--train_stage SFT_Test 
--SFT_actor_lora_r 16 
--SFT_load snap/ICR_SubMovie_Title64T_0_Q_Llama7b_SC/Epoch10_SFT
--FA2
```
### MMLU test of RLHF model
```shell
python MMLU.py
--seed 0 
--backbone snap/Llama-2-7b-hf 
--gpu cuda:1 
--max_token_length 2048 
--quantization 
--train_stage RLHF_Test 
--RLHF_actor_lora_r 4 
--RLHF_actor_lora_r 4 
--lm_head
--RLHF_load {xxx}
--FA2
```

# New


## SASRec Server start
```shell
cd SASRec/
python cli.py --dataset sub_movie --port 12621
```

## SFT stage

### SFT train
```shell
nohup accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 --main_process_port 29502 train.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/Llama-2-7b-hf-chat/ --item_index title64_t --batch_size 1 --topk 10 --clip_grad_norm 1.0 --epoch 40 --gen_max_length 512 --lr 0.0006 --gradient_accumulation_steps 12 --train_stage SFT --SFT_actor_lora_r 16 --warmup_ratio 0.0125 --val_batch_size 12 --SFT_train_tasks SFTSeqRec,SFTControlRec,SFTPersonalControlRec,SFTPersonalCategoryRate,SFTCategoryRate --SFT_val_tasks SFTTestSeqRec,SFTTestSeqRanking,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateLP_50,SFTTestItemCount --backup_ip 0.0.0.0 --val_epoch 5 --share_chat_gpt_ratio 0.5 --FA2 --llama2_chat_template --idx --SFT_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch07_SFT > snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/output8-40.log &
```

### SFT merge
```shell
python train.py --backbone snap/Llama-2-7b-hf-chat/ --gpu cuda:0 --train_stage SFT_Merge --SFT_actor_lora_r 16 --output snap/ICR_SubMovie_Title64T_0_Q_Llama7b/ --SFT_load snap/ICR_SubMovie_Title64T_0_Q_Llama7b/Epoch20_SFT
```


## RLHF stage

### RLHF train
```shell
python train.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch38/ --item_index title64_t --batch_size 8 --val_batch_size 12 --sample_num 2 --gradient_accumulation_steps 16 --topk 10 --clip_grad_norm 0.5 --epoch 4 --gen_max_length 512 --candidate_num 15 --quantization --train_stage RLHF --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_train_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate --RLHF_val_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate,RLHFItemCount --backup_ip 0.0.0.0 --lr 0.0000 --lora_drop 0.1 --weight_decay 0.01 --kl_coef 0.3 --entropy_weight 0.01 --vf_coef 0.1 --lm_head --policy_kl_threshold 0.1 --fine_grain_reward --idx --llama2_chat_template --FA2 --gpu cuda:0 --lr_power 4.0 --learn_batch 8 --new_data --sample_num 1

python train.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --item_index title64_t --batch_size 8 --val_batch_size 12 --sample_num 2 --gradient_accumulation_steps 16 --topk 10 --clip_grad_norm 0.5 --epoch 4 --gen_max_length 512 --candidate_num 15 --train_stage RLHF --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_train_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate --RLHF_val_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate,RLHFItemCount --backup_ip 0.0.0.0 --lr 0.0001 --lora_drop 0.1 --weight_decay 0.01 --kl_coef 0.3 --entropy_weight 0.01 --vf_coef 0.1 --lm_head --policy_kl_threshold 0.004 --fine_grain_reward --idx --llama2_chat_template --FA2 --gpu cuda:2 --lr_power 4.0 --learn_batch 8 --new_data --sample_num 1
```

### RLHF val
```shell
accelerate launch --num_processes 2 --gpu_ids 6,9 train.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch38/ --item_index title64_t --batch_size 8 --val_batch_size 24 --sample_num 2 --gradient_accumulation_steps 16 --topk 10 --clip_grad_norm 0.5 --epoch 4 --gen_max_length 512 --quantization --train_stage RLHF --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_train_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate --RLHF_val_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate,RLHFItemCount --backup_ip 0.0.0.0 --lr 0.0001 --lora_dropout 0.1 --weight_decay 0.01 --kl_coef 0.3 --entropy_weight 0.01 --vf_coef 0.1 --lm_head --policy_kl_threshold 0.1 --fine_grain_reward --idx --llama2_chat_template --FA2 --num_episodes 0 --dry --lr_power 4.0 --learn_batch 8 --candidate_num 15 --model_name RLHF_ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch38/Total_train_LM-True_FG-True_LR-0.0001_LDO-0.1_WD-0.01_KLC-0.3_EW-0.01_RS-False_RW-False_VFC-0.1_KLT-0.1_VM-False_LRP-4.0_GAS-16_LB-8_ND-True_NR-3_SN-1
```

### RLHF merge
```shell
python train.py --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-8_SN-2_Q-False_T3_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAS-4_LB-1_ND-False/ --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --item_index title64_t --gpu cuda:1 --train_stage RLHF_Merge --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-8_SN-2_Q-False_T3_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAS-4_LB-1_ND-False/3500step_RLHF --lm_head --FA2
```


### VLLM deploy
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --port 13579 --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/
```

### VLLM test
```shell
python task_test.py --SFT_test_task SFTTestSeqRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestSeqRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 5
python task_test.py --SFT_test_task SFTTestSeqRanking --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 5
python task_test.py --SFT_test_task SFTTestSeqRanking --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 3
python task_test.py --SFT_test_task SFT+TestPersonalControlRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFT-TestPersonalControlRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_30% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_50% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_70% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
```