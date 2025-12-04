"""
NOTE: allgather.py is for high-throughput. while low_latency_allgather.py is for low-latency.

Intel XPU version - adapted from NVIDIA CUDA implementation.

Key differences from NVIDIA version:
- Uses polling-based synchronization instead of CUDA stream wait/write APIs
- Uses torch.xpu.Stream instead of torch.cuda.Stream
- Inter-node communication uses torch.distributed instead of NVSHMEM
- No NVLink-specific optimizations (Intel uses different interconnects)
"""

import functools
import logging
import os
import time
from enum import Enum
from typing import List, Optional

import torch

from triton_dist.kernels.intel.common_ops import _set_signal_xpu, _wait_eq_xpu

# Setup logging for debugging
_log_level = os.environ.get("TRITON_DIST_LOG_LEVEL", "WARNING").upper()
_logger = logging.getLogger(__name__)


class RankLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that includes rank information."""
    def process(self, msg, kwargs):
        kwargs.setdefault('extra', {})['rank'] = self.extra.get('rank', '?')
        return msg, kwargs


def get_logger(rank: int = -1):
    """Get a logger with rank information."""
    return RankLoggerAdapter(_logger, {'rank': rank})


class AllGatherMethod(Enum):
    """AllGather communication methods."""
    Auto = 0
    All2All_IntraNode = 1
    All2All_InterNode = 2
    Ring1D_IntraNode = 3
    Ring2D_IntraNode = 4
    Ring1D_InterNode = 5
    Ring2D_InterNode = 6


@functools.lru_cache()
def get_numa_world_size_xpu() -> int:
    """
    Get NUMA world size for Intel XPU.

    TODO: Implement proper NUMA detection for Intel XPU.
    For now, assume all GPUs are in the same NUMA domain.
    """
    return int(torch.xpu.device_count())


@functools.lru_cache()
def get_auto_all_gather_method(num_ranks: int, num_local_ranks: int) -> AllGatherMethod:
    """
    Determine the best AllGather method for Intel XPU.

    Note: Intel XPU doesn't have NVLink, so we use different heuristics.
    """
    numa_world_size = get_numa_world_size_xpu()
    if num_local_ranks == num_ranks:
        # Single node
        if numa_world_size == num_ranks:
            return AllGatherMethod.Ring1D_IntraNode
        else:
            return AllGatherMethod.Ring2D_IntraNode
    else:
        # Multi-node
        return AllGatherMethod.Ring2D_InterNode


def sleep_async_xpu(duration_ms: int):
    """
    Async sleep for Intel XPU (for debugging purposes).

    Note: This is a simple CPU sleep, not a true GPU async sleep.
    Only used for correctness testing.
    """
    time.sleep(duration_ms / 1000.0)


def _add_noise_workload_debug():
    """Add random delay for debugging synchronization issues."""
    import random

    if random.random() > 0.3:
        sleep_async_xpu(100)


def cp_engine_producer_all_gather_full_mesh_push(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.xpu.Stream,
):
    """
    Full-mesh push AllGather for Intel XPU.

    Each rank pushes its data to all other ranks' buffers.
    """
    M_per_rank, N_per_rank = local_tensor.shape
    push_order = [(rank + i) % num_ranks for i in range(num_ranks)]
    src = local_tensor
    with torch.xpu.stream(stream):
        for dst_rank in push_order:
            dst = remote_tensor_buffers[dst_rank][rank * M_per_rank:(rank + 1) * M_per_rank, :]
            dst.copy_(src)
            # Set signal to indicate data is ready
            barrier_buffers[dst_rank][rank].fill_(1)


def cp_engine_producer_all_gather_full_mesh_pull(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.xpu.Stream,
    for_correctness: bool = False,
):
    """
    Full-mesh pull AllGather for Intel XPU.

    Each rank pulls data from all other ranks' buffers to its own buffer.
    """
    log = get_logger(rank)
    log.info(f"[full_mesh_pull] Starting: num_ranks={num_ranks}, tensor_shape={local_tensor.shape}")

    M_per_rank, _ = local_tensor.shape
    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.xpu.stream(stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            _add_noise_workload_debug()
        for idx, src_rank in enumerate(rank_orders):
            if src_rank == rank:
                log.info(f"[full_mesh_pull] Skipping self (src_rank={src_rank})")
                continue
            log.info(f"[full_mesh_pull] Copying from rank {src_rank} ({idx+1}/{num_ranks})")
            dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            dst.copy_(src)
            # Set signal to indicate data is ready
            barrier_buffers[rank][src_rank].fill_(1)
            log.info(f"[full_mesh_pull] Copied from rank {src_rank}, barrier set")
    log.info(f"[full_mesh_pull] Completed")


def cp_engine_producer_all_gather_ring_push_1d(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.xpu.Stream,
    for_correctness: bool = False,
):
    """
    1D Ring AllGather for Intel XPU.

    Each rank passes data to the next rank in a ring pattern.
    """
    log = get_logger(rank)
    log.info(f"[ring_push_1d] Starting: num_ranks={num_ranks}, tensor_shape={local_tensor.shape}")

    poll_count = [0]  # Use list for mutable in closure

    def wait_ready(rank: int, segment: int):
        """Wait for a segment to be ready (polling-based with proper sync)."""
        log.info(f"[ring_push_1d] wait_ready: waiting for rank={rank}, segment={segment}")
        # 必须同步当前 stream，确保能看到其他 GPU 的写入
        torch.xpu.synchronize()
        poll_count[0] = 0
        while barrier_buffers[rank][segment].item() != 1:
            poll_count[0] += 1
            if poll_count[0] % 10000 == 0:
                log.warning(f"[ring_push_1d] wait_ready: STILL waiting for rank={rank}, segment={segment}, poll_count={poll_count[0]}, barrier_value={barrier_buffers[rank][segment].item()}")
            # 添加小延迟避免过度轮询
            time.sleep(0.0001)
            torch.xpu.synchronize()  # 刷新缓存，重新读取
        log.info(f"[ring_push_1d] wait_ready: ready! rank={rank}, segment={segment}, poll_count={poll_count[0]}")

    def set_ready(rank: int, segment: int):
        """Set a segment as ready with proper synchronization."""
        log.info(f"[ring_push_1d] set_ready: setting rank={rank}, segment={segment}")
        # 确保之前的 copy 操作完成
        torch.xpu.synchronize()
        barrier_buffers[rank][segment].fill_(1)
        torch.xpu.synchronize()  # 确保信号写入完成
        log.info(f"[ring_push_1d] set_ready: done rank={rank}, segment={segment}")

    M_per_rank, _ = local_tensor.shape
    to_rank = (rank - 1 + num_ranks) % num_ranks
    log.info(f"[ring_push_1d] to_rank={to_rank}, M_per_rank={M_per_rank}")

    with torch.xpu.stream(stream):
        if for_correctness:
            # fake a slow communication case
            # test if the computation is waiting for the correct communication
            _add_noise_workload_debug()

        for stage in range(num_ranks - 1):
            send_segment = (rank + stage) % num_ranks
            M_start = send_segment * M_per_rank
            M_end = M_start + M_per_rank
            log.info(f"[ring_push_1d] Stage {stage}/{num_ranks-1}: send_segment={send_segment}, M_range=[{M_start}, {M_end})")
            if stage != 0:
                wait_ready(rank, send_segment)
            log.info(f"[ring_push_1d] Stage {stage}: copying data to rank {to_rank}")
            dst = remote_tensor_buffers[to_rank][M_start:M_end, :]
            src = remote_tensor_buffers[rank][M_start:M_end, :]
            dst.copy_(src)
            set_ready(to_rank, send_segment)
            log.info(f"[ring_push_1d] Stage {stage}: completed")
    log.info(f"[ring_push_1d] All stages completed")


def cp_engine_producer_all_gather_intra_node(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.xpu.Stream,
    for_correctness: bool = False,
    all_gather_method: AllGatherMethod = AllGatherMethod.All2All_IntraNode,
):
    """
    Intra-node AllGather dispatcher for Intel XPU.

    Selects the appropriate AllGather implementation based on the method.
    """
    log = get_logger(rank)
    log.info(f"[intra_node] Starting: method={all_gather_method}, num_ranks={num_ranks}, tensor_shape={local_tensor.shape}")
    log.info(f"[intra_node] remote_tensor_buffers count={len(remote_tensor_buffers)}, barrier_buffers count={len(barrier_buffers)}")

    if all_gather_method == AllGatherMethod.All2All_IntraNode:
        fn = cp_engine_producer_all_gather_full_mesh_pull
        log.info(f"[intra_node] Using full_mesh_pull")
    elif all_gather_method == AllGatherMethod.Ring1D_IntraNode:
        fn = cp_engine_producer_all_gather_ring_push_1d
        log.info(f"[intra_node] Using ring_push_1d")
    else:
        raise Exception(f"Unsupported allgather method: {all_gather_method}")

    log.info(f"[intra_node] Calling AllGather function")
    fn(
        rank,
        num_ranks,
        local_tensor,
        remote_tensor_buffers,
        barrier_buffers,
        stream,
        for_correctness=for_correctness,
    )
    log.info(f"[intra_node] AllGather function completed")


def cp_engine_producer_all_gather_ring_push_2d_inter_node(
    rank: int,
    num_local_ranks: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    intranode_stream: torch.xpu.Stream,
    internode_stream: Optional[torch.xpu.Stream] = None,
    for_correctness: bool = False,
):
    """
    2D Ring AllGather for inter-node communication on Intel XPU.

    Note: This is a simplified implementation. Full inter-node communication
    requires proper network fabric support (e.g., Intel MPI, OneCCL).

    Currently implements only the intra-node portion.
    Inter-node communication should be handled via torch.distributed.
    """
    def wait_ready(rank: int, segment: int):
        """Wait for a segment to be ready (polling-based)."""
        while barrier_buffers[rank][segment].item() != 1:
            pass

    def set_ready(rank: int, segment: int):
        """Set a segment as ready."""
        barrier_buffers[rank][segment].fill_(1)

    nnodes = num_ranks // num_local_ranks
    M_per_rank, N = local_tensor.shape
    local_rank = rank % num_local_ranks
    node_id = rank // num_local_ranks
    to_rank = (local_rank - 1 + num_local_ranks) % num_local_ranks

    with torch.xpu.stream(intranode_stream):
        if for_correctness:
            _add_noise_workload_debug()

        for n in range(nnodes):
            rank_base = ((n + node_id) % nnodes) * num_local_ranks
            if n != 0:  # inter node comm
                # For inter-node, use torch.distributed barrier
                torch.distributed.barrier()
                # Note: Actual inter-node data transfer should use
                # torch.distributed.all_gather or similar
                torch.distributed.barrier()

            # intra node comm
            for stage in range(num_local_ranks - 1):
                segment = (rank + stage) % num_local_ranks + rank_base
                M_start = segment * M_per_rank
                M_end = M_start + M_per_rank
                if stage != 0 or n != 0:
                    wait_ready(local_rank, segment)
                dst = remote_tensor_buffers[to_rank][M_start:M_end, :]
                src = remote_tensor_buffers[local_rank][M_start:M_end, :]
                dst.copy_(src)
                set_ready(to_rank, segment)


def cp_engine_producer_all_gather_full_mesh_pull_inter_node(
    rank: int,
    local_world_size: int,
    world_size: int,
    local_tensor: torch.Tensor,
    ag_buffer: List[torch.Tensor],
    signal_buffer: List[torch.Tensor],
    intranode_ag_stream: Optional[torch.xpu.Stream] = None,
    internode_ag_stream: Optional[torch.xpu.Stream] = None,
    signal_target: int = 1,
    for_correctness: bool = False,
):
    """
    Full-mesh pull AllGather for inter-node communication on Intel XPU.

    Note: This is a simplified implementation for Intel XPU.
    Inter-node communication uses torch.distributed primitives.
    """
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    M_per_rank, N = local_tensor.shape

    # Intra-node communication
    with torch.xpu.stream(intranode_ag_stream) if intranode_ag_stream else torch.xpu.device(local_tensor.device):
        # Copy local data to other local ranks
        for i in range(1, local_world_size):
            segment = rank * M_per_rank
            local_dst_rank = (local_rank + local_world_size - i) % local_world_size
            src = ag_buffer[local_rank][segment:segment + M_per_rank, :]
            dst = ag_buffer[local_dst_rank][segment:segment + M_per_rank, :]
            dst.copy_(src)
            _set_signal_xpu(signal_buffer[local_dst_rank][rank], signal_target)

        # For inter-node, we need to wait for remote data and then distribute locally
        for i in range(1, n_nodes):
            recv_rank = (local_rank + (n_nodes - i) % n_nodes * local_world_size)
            recv_segment = recv_rank * M_per_rank
            # Wait for inter-node data (would need proper inter-node sync)
            _wait_eq_xpu(signal_buffer[local_rank][recv_rank], signal_target)
            src = ag_buffer[local_rank][recv_segment:recv_segment + M_per_rank, :]
            for j in range(1, local_world_size):
                local_dst_rank = (local_rank + local_world_size - j) % local_world_size
                dst = ag_buffer[local_dst_rank][recv_segment:recv_segment + M_per_rank, :]
                dst.copy_(src)
                _set_signal_xpu(signal_buffer[local_dst_rank][recv_rank], signal_target)

    # Synchronize streams if both are provided
    if intranode_ag_stream and internode_ag_stream:
        intranode_ag_stream.wait_stream(internode_ag_stream)


def cp_engine_producer_all_gather_inter_node(
    local_tensor: torch.Tensor,
    ag_buffer: List[torch.Tensor],
    signal_buffer: List[torch.Tensor],
    signal_target: int,
    rank: int,
    local_world_size: int,
    world_size: int,
    intranode_ag_stream: Optional[torch.xpu.Stream] = None,
    internode_ag_stream: Optional[torch.xpu.Stream] = None,
    for_correctness: bool = False,
    all_gather_method: AllGatherMethod = AllGatherMethod.All2All_InterNode,
):
    """
    Inter-node AllGather dispatcher for Intel XPU.

    Note: Inter-node communication on Intel XPU typically uses OneCCL
    through torch.distributed. This implementation provides the framework
    but actual inter-node data movement should use torch.distributed.all_gather.
    """
    if all_gather_method == AllGatherMethod.All2All_InterNode:
        cp_engine_producer_all_gather_full_mesh_pull_inter_node(
            rank,
            local_world_size,
            world_size,
            local_tensor,
            ag_buffer,
            signal_buffer,
            intranode_ag_stream,
            internode_ag_stream,
            signal_target=signal_target,
            for_correctness=for_correctness,
        )
    elif all_gather_method == AllGatherMethod.Ring2D_InterNode:
        cp_engine_producer_all_gather_ring_push_2d_inter_node(
            rank,
            local_world_size,
            world_size,
            local_tensor,
            ag_buffer,
            signal_buffer,
            intranode_ag_stream,
            None,
            for_correctness=for_correctness,
        )
    else:
        raise Exception(f"Unsupported allgather method: {all_gather_method}")
