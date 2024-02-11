python train.py
--backbone /home/wangshuo/weights/llama2/Llama-2-7b-hf-chat/
--gpu cuda:1
--train_stage SFT_Merge
--SFT_actor_lora_r 16
--output /home/wangshuo/codes/InstructControllableRec_RLHF/snap/0207-steam-single/test-epoch32/
--SFT_load /home/wangshuo/codes/InstructControllableRec_RLHF/snap/0207-steam-single/Epoch32_SFT