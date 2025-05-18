import torch
import multiprocessing as mp
import sys
from vllm.config import CacheConfig, VllmConfig, KVTransferConfig

import warnings

from cpu_memory_connector import (
    SharedCPUMemoryConnector,
    ReqMeta,
    SharedCPUMemmoryConnectorMetadata,
    KVConnectorRole,
    align_to_block_size
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_kv_cache_insert_retrieve():
    # Create a VllmConfig with necessary settings for testing
    cache_config = CacheConfig(block_size=4)
    kv_transfer_config = KVTransferConfig()
    vllm_config = VllmConfig(
        cache_config=cache_config, 
        kv_transfer_config=kv_transfer_config
    )
    connector = SharedCPUMemoryConnector(
        vllm_config, 
        KVConnectorRole.WORKER
    )
    # Create test data
    layer_name = "layer_0"
    token_ids = torch.tensor([1, 2, 3, 4])
    # Create a sample KV cache tensor
    kv_cache = torch.randn(2, 8)  # Shape can vary based on actual implementation
    
    # Generate a key for the KV cache
    key = connector._generate_key_debug(layer_name, token_ids)
    
    # Insert the KV cache into the map
    connector._insert_into_shared_memory(key, kv_cache)

    # use multiprocessing to spwan a process to retrive the kv cache
    def retrieve_kv_cache(key, shape, dtype, vllm_config):
        # Create a new connector in the child process
        child_connector = SharedCPUMemoryConnector(
            vllm_config,
            KVConnectorRole.WORKER
        )
        # make a new tensor shaped as shape
        kv_cache_return = child_connector._load_from_shared_memory(key, shape,dtype)
        # assert that the retrieved tensor is equal to the original kv_cache
        assert torch.testing.assert_close(kv_cache_return, kv_cache), "Retrieved KV cache does not match the original"

    # Start the process and get the result
    process = mp.Process(target=retrieve_kv_cache, args=(key, kv_cache.shape, kv_cache.dtype, vllm_config))
    process.start()
    process.join()
    connector._clear_shared_memory(key)