import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from multiprocessing import shared_memory
import warnings

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


# 过滤shared_memory泄漏警告
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message="resource_tracker: There appear to be .* leaked shared_memory objects to clean up at shutdown",
    module="multiprocessing.resource_tracker"
)

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool

    @staticmethod
    def make_meta(
        token_ids: list[int], block_ids: list[int], block_size: int, is_store: bool
    ) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        # 在一个分块的 block 表上，但是最后会 flatten 成一个一维的 index selector.
        # block_size = 4
        # block_offsets = [[0,1,2,3]]
        # block_ids = [1,3,5,6]
        # block_ids.reshape((num_blocks, 1)) = [[1], [3], [5], [6]]
        # block_ids.reshape((num_blocks, 1)) * block_size = [[1*4], [3*4], [5*4], [6*4]]
        # [ [0,1,2,3] + [1*4], [0,1,2,3] + [3*4], [0,1,2,3] + [5*4], [0,1,2,3] + [6*4]]
        # 利用广播加法的性质
        # 基于 block id 和 block_size 转换成在一个连续内存的小标索引矩阵然后再flatten。
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
        )


@dataclass
class SharedCPUMemmoryConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store)
        )


class SharedCPUMemmoryConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the shared memory.
    # It does not support the chunk prefill.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        print("init")
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        # requests that need to load the KV cache
        self._requests_need_load: dict[str, Request] = {}
        self._request_kvcache_map = {}  # layer.hash(input_ids) -> cpu kv cache
        self._request_shm_map = {}  # layer.hash(input_ids) -> shm object
        logger.info(vllm_config.kv_transfer_config)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        attn_metadata = forward_context.attn_metadata

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1
                )
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1
                )
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, SharedCPUMemmoryConnectorMetadata)

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("In connector.start_load_kv, but the attn_metadata is None")
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue
            logger.info(
                "Inject KV cache of %d tokens to the paged memory",
                len(request.slot_mapping),
            )
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[forward_context.virtual_engine]
                # 这里应该是默认不开chunk prefill的，每一层的prefill算完就存下来。
                key = self._generate_key_debug(layer_name, request.token_ids)
                kv_cache = self._request_kvcache_map[key].cuda()
                inject_kv_into_layer(kv_cache_layer, kv_cache, request.slot_mapping)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, SharedCPUMemmoryConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                key = self._generate_key_debug(layer_name, request.token_ids)
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)

                self._request_kvcache_map[key] = kv_cache.detach().cpu()

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        if not self._found_match_for_request(request):
            return 0, False

        logger.info("External Cache Hit!")

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size
        )

        return num_tokens_to_check - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = SharedCPUMemmoryConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                )
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                if not self._found_match_for_request(new_req):
                    meta.add_request(
                        token_ids=new_req.prompt_token_ids,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        is_store=True,
                    )

        for cached_req in scheduler_output.scheduled_cached_reqs:
            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not cached_req.resumed_from_preemption:
                break
            if cached_req.req_id in self._requests_need_load:
                # NOTE(rob): cached_req_data does not have the full
                # list of token ids (only new tokens). So we look it
                # up in the actual request object.
                request = self._requests_need_load[cached_req.req_id]
                total_tokens = (
                    len(cached_req.new_token_ids) + cached_req.num_computed_tokens
                )
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = cached_req.new_block_ids[0]

                meta.add_request(
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                    is_store=False,
                )
                total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    # =========
    # shared memory management functions
    # =========
    def _insert_into_shared_memory(self, key: str, kv_cache: torch.Tensor):
        """Insert the KV cache into the shared memory."""
        kv_cache_np = kv_cache.cpu().numpy()
        # Create a shared memory block
        shm = shared_memory.SharedMemory(
            name=key, create=True, size = kv_cache_np.nbytes
        )
        shm_array = np.ndarray(kv_cache_np.shape, dtype=kv_cache_np.dtype, buffer=shm.buf)
        # Copy the data to the shared memory
        shm_array[:] = kv_cache_np[:]
        self._request_kvcache_map[key] = shm_array
        self._request_shm_map[key] = shm

    def _load_from_shared_memory(self, key: str, shape: tuple[int], dtype: torch.dtype) -> torch.Tensor:
        """Load the KV cache from the shared memory."""
        # Create a shared memory block
        shm = shared_memory.SharedMemory(name=key)
        # Create a numpy array from the shared memory
        shm_array = np.ndarray(
            shape, dtype=dtype.numpy_dtype, buffer=shm.buf
        )
        # Copy the data to the shared memory, kv has the same memory with shm_array
        kv_cache = torch.from_numpy(shm_array)
        return kv_cache

    def _clear_shared_memory(self, key: str):
        """Clear the shared memory."""
        shm = shared_memory.SharedMemory(name=key)
        shm.close()
        shm.unlink()
        del self._request_kvcache_map[key]
        del self._request_shm_map[key]

    # ==============================
    # Helper functions
    # ==============================
    def _found_match_for_request(
        self,
        request: "Request",
    ) -> bool:
        """Check if the cache is hit for the request."""
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size
        )
        token_ids_key = self._generate_token_ids_key_debug(
            torch.tensor(request.prompt_token_ids)[:num_tokens_to_check]
        )
        # how to ensure all the layers exists?
        return token_ids_key in self._request_kvcache_map

    def _generate_token_ids_key_debug(
        self,
        input_ids: torch.Tensor,
    ) -> str:
        """Generate a name based on the hash of the bytes of the input
        ids.
        """
        input_ids_bytes = input_ids.numpy().tobytes()
        input_ids_hash = hashlib.md5(input_ids_bytes, usedforsecurity=False).hexdigest()
        return input_ids_hash

    def _generate_key_debug(
        self,
        layer_name: str,
        input_ids: torch.Tensor,
    ) -> str:
        """Generate a key based on the layer name and the hash
        of the bytes of the input ids.
        """
        token_ids_key = self._generate_token_ids_key_debug(input_ids)
        return f"{token_ids_key}.{layer_name}"


def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size."""
    return (num_tokens - 1) // block_size * block_size
