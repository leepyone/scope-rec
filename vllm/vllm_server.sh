CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
--model /home/wangshuo/weights/llama2/Llama-2-7b-hf-chat/ \
--port 8010 >./server_log/0415-steam.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.api_server \