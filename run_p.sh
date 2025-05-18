#!/bin/bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
    --port 8100 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"SharedCPUMemoryConnector","kv_rank":0,"kv_parallel_size":2,"kv_role":"kv_producer","kv_connector_module_path":"cpu_memory_connector"}'
    