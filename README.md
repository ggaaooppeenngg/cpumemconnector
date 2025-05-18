# Shared CPU Memory Connector for KV Transfer

This project demonstrates how to implement an external connector for KV (Key-Value) transfer in **vLLM**, using shared CPU memory. It's particularly useful for scenarios where GPU memory is limited and cannot hold the transfer buffer.

This connector is adapted from the `SharedStorageMemoryConnector` provided by vLLM.

## Why Shared CPU Memory?

By utilizing CPU shared memory for KV transfer, this approach:

* Enables memory sharing between processes without requiring additional GPU memory.
* Supports low-GPU-memory environments while maintaining performance.

## Setup

### Prefill Node

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
    --port 8200 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{
        "kv_connector": "SharedCPUMemoryConnector",
        "kv_rank": 0,
        "kv_parallel_size": 2,
        "kv_role": "kv_producer",
        "kv_connector_module_path": "cpu_memory_connector"
    }'
```

### Decode Node

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-0.6B \
    --port 8200 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{
        "kv_connector": "SharedCPUMemoryConnector",
        "kv_rank": 1,
        "kv_parallel_size": 2,
        "kv_role": "kv_consumer",
        "kv_connector_module_path": "cpu_memory_connector"
    }'
```

## Notes

* Ensure that both the prefill and decode nodes are configured consistently, particularly in terms of `kv_parallel_size`.
* The `kv_rank` should be unique for each process (e.g., `0` for producer, `1` for consumer).
* This connector must be placed in a Python module or script accessible via the `PYTHONPATH`.

---

Let me know if you'd like to add diagrams or examples of the connector implementation.
