################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import argparse
import os
import sys

import functools

# import nvshmem.core
import torch

import triton
import triton.language as tl

# from triton.language.extra.cuda.language_extra import (__syncthreads, atomic_add, atomic_cas, ld, ld_acquire, st, tid)


# import triton_dist.language as dl

# from triton_dist.autotuner import contextual_autotune
# from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
# from triton_dist.utils import (assert_allclose, dist_print, group_profile, perf_func)

# ==============================================================================================================
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

from torch import Tensor


import torch.distributed._symmetric_memory as symm_mem

from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from torch._C._distributed_c10d import _SymmetricMemory

import datetime
import numpy as np
import random
import logging
# ==============================================================================================================

ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func
        return func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", default=False)
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="LLaMA-3.1-70B", choices=configs.keys())
    parser.add_argument("--debug", action="store_true", default=False)
    # parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
    #                     default=torch.xpu.get_device_capability() >= (9, 0))
    parser.add_argument("--persistent", action="store_true",
                        default=False)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--local_world_size", default=8, type=int)

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")

# ===========================================================================================================
# _mr_references = {}

# _TP_GROUP = None

_group_name_to_workspace_tensor: dict[str, torch.Tensor | None] = {}

def create_local_tensor(shape, dtype) -> torch.Tensor:
    torch.xpu.synchronize()
    tensor = symm_mem.empty(shape, dtype=dtype, device=torch.device(f"xpu:{torch.xpu.current_device()}"))
    torch.xpu.synchronize()
    return tensor


def create_symmetric_handle(shape, dtype, group_name) -> _SymmetricMemory:
    # group_name = _TP_GROUP.group_name
    enable_symm_mem_for_group(group_name)
    tensor = _group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    req_size = 1
    for s in shape:
        req_size *= s
    req_size *= torch.empty([], dtype=dtype).element_size()
    if tensor is None or size < req_size:
        # need calculate stride ?
        tensor = _SymmetricMemory.empty_strided_p2p(
            shape,
            torch._prims_common.make_contiguous_strides_for(shape),
            dtype,
            torch.device(f"xpu:{torch.xpu.current_device()}"),
            group_name,
        )
        _group_name_to_workspace_tensor[group_name] = tensor
    return _SymmetricMemory.rendezvous(tensor)


def get_remote_tensors(local_sm_handle, local_tensor, rank, local_world_size) -> List[torch.Tensor]:
    def _get_peer_tensor(t, peer) -> torch.Tensor:
        if peer == rank:
            return t
        return local_sm_handle.get_remote_tensor(peer, tuple(t.size()), t.dtype)

    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    rank_on_same_node_end = rank_on_same_node_start + local_world_size
    return [_get_peer_tensor(local_tensor, peer) for peer in range(rank_on_same_node_start, rank_on_same_node_end)]

    # def _get_peer_tensor(t, peer) -> torch.Tensor:
    #     # avoid create tensor on the same buf again. nvshmem4py can't handle multiple reference with grace. so we handle it here.
    #     # https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-finalize-does-not-handle-everything/337979
    #     if peer == rank:
    #         return t
    #     return nvshmem_core_get_peer_tensor(t, peer)

    # local_rank = rank % local_world_size
    # rank_on_same_node_start = rank - local_rank
    # rank_on_same_node_end = rank_on_same_node_start + local_world_size
    # torch.cuda.synchronize()
    # tensor = nvshmem_create_tensor(shape, dtype=dtype)
    # torch.cuda.synchronize()
    # return [_get_peer_tensor(tensor, peer) for peer in range(rank_on_same_node_start, rank_on_same_node_end)]

def get_remote_signal_pad(local_sm_handle, rank, local_world_size) -> List[torch.Tensor]:
    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    rank_on_same_node_end = rank_on_same_node_start + local_world_size
    return [local_sm_handle.get_signal_pad(peer) for peer in range(rank_on_same_node_start, rank_on_same_node_end)]

class AllGatherMethod(Enum):
    Auto = 0
    All2All_IntraNode = 1
    All2All_InterNode = 2
    Ring1D_IntraNode = 3
    Ring2D_IntraNode = 4
    Ring1D_InterNode = 5
    Ring2D_InterNode = 6

@dataclass
class AllGatherGEMMTensorParallelContext:
    # problem size
    # local input [M_per_rank, K]
    # local weight [K, N_per_rank]
    max_M: int
    N_per_rank: int
    K: int
    tensor_dtype: torch.dtype
    # parallelism info
    rank: int
    num_ranks: int
    num_local_ranks: int
    is_multinode: bool = field(init=False)
    n_nodes: int = field(init=False)
    node_rank: int = field(init=False)
    local_rank: int = field(init=False)
    symm_workspaces: List[torch.Tensor] = field(init=False)  # ag buffer
    symm_barriers: List[torch.Tensor] = field(init=False)
    symm_workspace: torch.Tensor = field(init=False)
    symm_barrier: torch.Tensor = field(init=False)
    fake_barrier: torch.Tensor = field(init=False)  # for gemm only function
    symm_comm_buf: torch.Tensor = field(init=False)
    barrier_target = 1
    # async streams
    ag_intranode_stream: Optional[torch.xpu.streams.Stream] = None
    ag_internode_stream: Optional[torch.xpu.streams.Stream] = None
    # triton compute kernel config
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_SIZE_M: int = 8
    stages: int = 3
    warps: int = 8
    max_blocks: int = 1  # max number of blocks on GPU used by customized barrier_all
    max_gemm_sm: int = field(init=False)
    phase: int = 1
    all_gather_method: AllGatherMethod = AllGatherMethod.Auto
    # testing options
    for_correctness: bool = False

    def __post_init__(self):
        assert self.num_ranks % self.num_local_ranks == 0
        self.is_multinode = self.num_ranks > self.num_local_ranks
        self.n_nodes = self.num_ranks // self.num_local_ranks
        self.node_rank = self.rank // self.num_local_ranks
        self.local_rank = self.rank % self.num_local_ranks

        ####### create symmetric mem workspace
        self.symm_workspaces_group = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="xccl")
        torch.distributed.barrier(self.symm_workspaces_group)

        group_name = self.symm_workspaces_group.group_name
        self.symm_workspaces_handle = create_symmetric_handle((self.max_M, self.K), self.tensor_dtype, group_name)
        ####### get remote workspace tensors
        self.symm_workspaces = get_remote_tensors(self.symm_workspaces_handle, _group_name_to_workspace_tensor[group_name], self.rank,
                                                      self.num_local_ranks)
        self.symm_workspace = self.symm_workspaces[self.local_rank]

        self.symm_barriers = get_remote_signal_pad(self.symm_workspaces_handle, self.rank, self.num_local_ranks)
        self.symm_barrier = self.symm_barriers[self.local_rank]

        ####### create symmetric mem for syncing
        self.symm_comm_bufs_group = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="xccl")
        torch.distributed.barrier(self.symm_comm_bufs_group)

        group_name_sync = self.symm_comm_bufs_group.group_name
        self.symm_comm_bufs_handle = create_symmetric_handle((3 * self.num_ranks, ), torch.int32, group_name_sync)
        ####### get remote workspace tensors
        self.symm_comm_bufs = get_remote_tensors(self.symm_comm_bufs_handle, _group_name_to_workspace_tensor[group_name_sync], self.rank,
                                                      self.num_local_ranks)
        self.symm_comm_buf = self.symm_comm_bufs[self.local_rank]
        # self.symm_comm_buf.fill_(0)

        # barrier_dtype = NVSHMEM_SIGNAL_DTYPE if self.is_multinode else torch.int32
        # single node for now
        barrier_dtype = torch.int32

        ####### replaced by signal pad in symmetric mem
        # self.symm_barriers_group = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="xccl")
        # torch.distributed.barrier(self.symm_barriers_group)
        # self.symm_barriers = nvshmem_create_tensors((self.num_ranks, ), barrier_dtype, self.rank, self.num_local_ranks,
        #                                             self.symm_barriers_group.group_name)
        # self.symm_barrier = self.symm_barriers[self.local_rank]
        # self.symm_barrier.fill_(0)

        self.fake_barrier = torch.ones([self.num_ranks], dtype=barrier_dtype, device="xpu")
        self.max_gemm_sm = torch.xpu.get_device_properties("xpu").gpu_eu_count

        # nvshmem_barrier_all_on_stream(torch.xpu.current_stream())
        torch.xpu.synchronize()

    def finailize(self):
        # nvshmem_free_tensor_sync(self.symm_workspace)
        # nvshmem_free_tensor_sync(self.symm_barrier)
        # nvshmem_free_tensor_sync(self.symm_comm_buf)
        import torch.distributed as dist
        for _, g in _group_name_to_workspace_tensor.items():
            dist.destroy_process_group(group=g)

        del _group_name_to_workspace_tensor

@functools.lru_cache()
def get_numa_world_size():
    # return torch.xpu.device_count()
    return 2

@functools.lru_cache()
def get_auto_all_gather_method(num_ranks, num_local_ranks):
    # if has_fullmesh_nvlink():
    #     if num_ranks == num_local_ranks:
    #         return AllGatherMethod.All2All_IntraNode
    #     else:
    #         return AllGatherMethod.All2All_InterNode
    # else:
    numa_world_size = get_numa_world_size()
    if num_local_ranks == num_ranks:
        if numa_world_size == num_ranks:
            return AllGatherMethod.Ring1D_IntraNode
        else:
            return AllGatherMethod.Ring2D_IntraNode
    else:
        return AllGatherMethod.Ring2D_InterNode

def create_ag_gemm_context(tensor_A, tensor_B, rank, num_ranks, max_M, num_local_ranks=8, BLOCK_M=128, BLOCK_N=256,
                           BLOCK_K=64, stages=3, ag_intranode_stream=None, ag_internode_stream=None,
                           for_correctness=False):
    """create context for allgather gemm intra-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        rank (int): current rank
        num_ranks (int): total number of ranks
        max_M: max number of M shape, should be greater than M_per_rank * num_ranks
        max_blocks: max number of blocks on GPU
        BLOCK_M (int, optional): GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): GEMM tiling factor for K dim. Defaults to 64.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        ag_intranode_stream (torch.cuda.streams.Stream, optional): The stream used for intranode communication of allgather, if not provided, create a new one. Defaults to None.
        ag_internode_stream (torch.cuda.streams.Stream, optional): The stream used for internode communication of allgather, if not provided, create a new one. Defaults to None.
        for_correctness (bool, optional): if only for correctness, communication would sleep some seconds to
            trigger possible synchronization and dependency bugs. Defaults to False.

    Returns:
        AllGatherGEMMTensorParallelContext
    """
    M_per_rank, K = tensor_A.shape
    N_per_rank, _ = tensor_B.shape
    assert tensor_A.shape[1] == tensor_B.shape[
        1], f"tensor_B should has shape (col_major) [{N_per_rank}, {K}], but get [{tensor_B.shape}]"
    assert tensor_A.dtype == tensor_B.dtype, f"Dtype of input and weight must be same: tensor_A dtype {tensor_A.dtype}, tensor_B dtype {tensor_B.dtype}"

    dtype = tensor_A.dtype
    ag_intranode_stream = torch.xpu.Stream() if ag_intranode_stream is None else ag_intranode_stream
    ag_internode_stream = torch.xpu.Stream() if ag_internode_stream is None else ag_internode_stream

    ctx = AllGatherGEMMTensorParallelContext(
        N_per_rank=N_per_rank, K=K, tensor_dtype=dtype, rank=rank, num_ranks=num_ranks, num_local_ranks=num_local_ranks,
        max_M=max_M, ag_intranode_stream=ag_intranode_stream, ag_internode_stream=ag_internode_stream, BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, stages=stages,
        all_gather_method=get_auto_all_gather_method(num_ranks, num_local_ranks), for_correctness=for_correctness)

    # nvshmem_barrier_all_on_stream()
    torch.xpu.synchronize()
    return ctx

def init_seed(seed=0):
    # os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    # zero empty takes more kernel launch and may hide uninitialized problem. always set to False
    # available since torch 2.2: https://docs.pytorch.org/docs/2.2/deterministic.html
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:
        logging.warning("torch.utils.fill_uninitialized_memory is available only for torch >=2.2")
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.xpu.manual_seed_all(3 + seed)
    # torch.backends.mkldnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)

def initialize_distributed(seed=None):
    # global _TP_GROUP
    # assert _TP_GROUP is None, "TP_GROUP has already been initialized"

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.xpu.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="cpu:gloo,xpu:xccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    # _TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="xccl")
    # print("xccl group: ", _TP_GROUP)
    # torch.distributed.barrier(_TP_GROUP)
    # _TP_GROUP_GLOO = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")
    # print("gloo group: ", _TP_GROUP_GLOO)
    # torch.distributed.barrier(_TP_GROUP_GLOO)

    init_seed(seed=seed if seed is not None else RANK)
    # init_nvshmem_by_torch_process_group(_TP_GROUP_GLOO)
    # return _TP_GROUP

# @triton.jit
# def _is_gpu_master():
#     pid_x = tl.program_id(axis=0)
#     pid_y = tl.program_id(axis=1)
#     pid_z = tl.program_id(axis=2)
#     return (pid_x + pid_y + pid_z) == 0

# @triton.jit
# def _is_cta_master():
#     thread_idx_x = tid(0)
#     thread_idx_y = tid(1)
#     thread_idx_z = tid(2)
#     return (thread_idx_x + thread_idx_y + thread_idx_z) == 0

# @triton.jit
# def unsafe_barrier_on_this_grid(ptr):
#     """ triton implementation of cooperative_group::thid_grid().sync()
#     WARNING: use with care. better launch triton with launch_cooperative_grid=True to throw an explicit error instead of hang without notice.
#     """
#     __syncthreads()
#     pid_size_x = tl.num_programs(axis=0)
#     pid_size_y = tl.num_programs(axis=1)
#     pid_size_z = tl.num_programs(axis=2)
#     expected = pid_size_x * pid_size_y * pid_size_z
#     if _is_cta_master():
#         nb = tl.where(
#             _is_gpu_master(),
#             tl.cast(0x80000000, tl.uint32, bitcast=True) - (expected - 1),
#             1,
#         )
#         old_arrive = atomic_add(ptr.to(tl.pointer_type(tl.uint32)), nb, scope="gpu", semantic="release")
#     else:
#         old_arrive = tl.cast(0, tl.uint32)

#     if _is_cta_master():
#         current_arrive = ld_acquire(ptr)
#         while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
#             current_arrive = ld_acquire(ptr, scope=tl.constexpr("gpu"))

#     __syncthreads()

# @triton.jit
# def barrier_on_this_grid(ptr, use_cooperative: tl.constexpr):
#     # if use_cooperative:
#     #     cooperative_barrier_on_this_grid()
#     # else:
#     unsafe_barrier_on_this_grid(ptr)

@triton.jit(do_not_specialize=["rank"])
def copy_kernel(
    rank,
    local_buf_ptr,
    global_buf_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_sms = tl.num_programs(axis=0)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    num_iters_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_iters_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_iters = num_iters_m * num_iters_n

    for i in range(sm_id, num_iters, num_sms):
        pid_m = i // num_iters_n
        pid_n = i % num_iters_n
        data_ptr = local_buf_ptr + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m + (
            pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
        dst_ptr = global_buf_ptr + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m + (
            pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
        mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)
        mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (pid_n * BLOCK_SIZE_N + offs_n[None, :] < N)

        data = tl.load(data_ptr, mask=mask_data)
        tl.store(dst_ptr, data, mask=mask_dst)

@triton.jit
def _barrier_all_intra_node_atomic_cas_once(local_rank, rank, local_world_size, symm_flags_ptrs):
    """
    Intel Triton implementation of intra-node barrier using atomic CAS.
    Uses IPC-based symmetric memory pointers for cross-rank synchronization.

    symm_flags_ptrs: pointer to an array of pointers, each pointing to the symmetric flag buffer of each rank
    """
    # Phase 1: Signal all other ranks that we have arrived
    # Each rank writes to its own slot in every other rank's buffer
    for i in range(local_world_size):
        remote_base_ptr = tl.load(symm_flags_ptrs + i).to(tl.pointer_type(tl.int32))
        # Try to set remote_ptr[local_rank] from 0 to 1
        while tl.atomic_cas(remote_base_ptr + local_rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    # Phase 2: Wait for all other ranks to signal us
    # Check our own buffer for signals from all ranks
    local_base_ptr = tl.load(symm_flags_ptrs + local_rank).to(tl.pointer_type(tl.int32))
    for i in range(local_world_size):
        # Wait until local_ptr[i] becomes 1, then reset to 0
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "target_value"])
def barrier_all_intra_node_non_atomic(local_rank, rank, num_ranks, symm_flags_ptrs, target_value,
                                      use_cooperative: tl.constexpr):
    """
    Intel Triton implementation of intra-node barrier.

    Since Intel Triton doesn't support cooperative grid launch, we use atomic CAS
    based barrier which works with IPC symmetric memory.

    symm_flags_ptrs: pointer to an array of pointers for IPC-based barrier
    """
    # Only the first program (work-group) performs the barrier
    pid = tl.program_id(axis=0)
    if pid == 0:
        _barrier_all_intra_node_atomic_cas_once(local_rank, rank, num_ranks, symm_flags_ptrs)


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "flag_value"])
def copy_and_barrier_all_intra_node_kernel(
    local_rank,
    rank,
    num_ranks,
    local_buf_ptr,
    global_buf_ptr,
    symm_barrier_ptr,
    symm_sync_ptrs,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    flag_value,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    """
    Intel Triton implementation of copy + barrier kernel.

    This kernel:
    1. Performs a barrier to synchronize all ranks before copy
    2. Copies local data to global buffer
    3. Sets the symm_barrier signal
    4. Performs another barrier after copy

    Note: use_cooperative is ignored on Intel (always False behavior)
    """
    # Pre-copy barrier: ensure all ranks are ready
    barrier_all_intra_node_non_atomic(local_rank, rank, num_ranks, symm_sync_ptrs, flag_value, use_cooperative)

    # Copy local data to global buffer
    copy_kernel(rank, local_buf_ptr, global_buf_ptr, M_per_rank, N, stride_local_m, stride_local_n, stride_global_m,
                stride_global_n, BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Set symm barrier signal - only first program does this
    pid = tl.program_id(axis=0)
    if pid == 0:
        # Set barrier_ptr[rank] = 1, others = 0
        for i in range(num_ranks):
            if i == rank:
                tl.store(symm_barrier_ptr + i, 1)
            else:
                tl.store(symm_barrier_ptr + i, 0)

    # Post-copy barrier: ensure all ranks have completed copy
    barrier_all_intra_node_non_atomic(local_rank, rank, num_ranks, symm_sync_ptrs, flag_value + 1, use_cooperative)

def local_copy_and_barrier_all(local_rank, rank, num_ranks, local_data, global_data, comm_bufs, barrier_ptr, M_per_rank,
                               N, phase, is_internode: bool = False, use_cooperative: bool = False): # disable cooperative grid since intel triton doesn't support it
    # if not is_internode:
    grid = lambda META: (min(
        triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        torch.xpu.get_device_properties("xpu").gpu_eu_count), )
    additional_options = {}
    CPY_BLOCKS = [128, 256]

    # if use_cooperative:
    #     additional_options.update(launch_cooperative_grid_options())
    copy_and_barrier_all_intra_node_kernel[grid](local_rank, rank, num_ranks, local_data,
                                                    global_data, barrier_ptr, comm_bufs, M_per_rank, N,
                                                    local_data.stride(0), local_data.stride(1), global_data.stride(0),
                                                    global_data.stride(1), phase, CPY_BLOCKS[0], CPY_BLOCKS[1],
                                                    use_cooperative, **additional_options)
    # else:
    #     nvshmem_barrier_all_on_stream()
    #     barrier_ptr.fill_(0)
    #     grid = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    #     copy_kernel[grid](rank, local_data, global_data, M_per_rank, N, local_data.stride(0), local_data.stride(1),
    #                       global_data.stride(0), global_data.stride(1), 128, 256)
    #     _set_signal_cuda(barrier_ptr[rank], 1, torch.cuda.current_stream())
    #     nvshmem_barrier_all_on_stream()

# def _kernel_consumer_gemm_non_persistent_repr(proxy):
#     constexprs = proxy.constants
#     cap_major, cap_minor = torch.cuda.get_device_capability()
#     a_dtype = proxy.signature["a_ptr"].lstrip("*")
#     b_dtype = proxy.signature["b_ptr"].lstrip("*")
#     c_dtype = proxy.signature["c_ptr"].lstrip("*")
#     BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]
#     if constexprs.get("stride_am", None) == 1:  # column major => n
#         a_trans = "n"
#     elif constexprs.get("stride_ak", None) == 1:  # row-major => t
#         a_trans = "t"
#     else:
#         raise Exception("both stride_am/stride_ak != 1")

#     if constexprs.get("stride_bk", None) == 1:
#         b_trans = "n"
#     elif constexprs.get("stride_bn", None) == 1:
#         b_trans = "t"
#     else:
#         raise Exception("both stride_am/stride_ak != 1")

#     if constexprs.get("stride_cm", None) == 1:
#         c_trans = "n"
#     elif constexprs.get("stride_cn", None) == 1:
#         c_trans = "t"
#     else:
#         raise Exception("both stride_am/stride_ak != 1")

#     return f"triton3x_sm{cap_major}{cap_minor}_ag_gemm_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(do_not_specialize=["rank"], launch_metadata=_matmul_launch_metadata,
            )
def kernel_consumer_gemm_non_persistent(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn, rank, WORLD_SIZE: tl.constexpr, barrier_ptr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    a_dtype = a_ptr.dtype.element_ty
    b_dtype = b_ptr.dtype.element_ty
    c_dtype = c_ptr.dtype.element_ty
    # IS_FP8 = tl.constexpr(a_dtype == tl.float8e5) or tl.constexpr(a_dtype == tl.float8e4nv)
    tl.static_assert(a_dtype == b_dtype, "A and B must have the same dtype")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # threadblock swizzle
    #  no stream-k support. only split by m x n
    m_per_rank = M // WORLD_SIZE
    m_offset = m_per_rank * rank
    pid_m_offset = tl.cdiv(m_offset, BLOCK_SIZE_M)
    pid_m = (pid_m + pid_m_offset) % num_pid_m

    # wait for segment ready.
    offs_am = pid_m * BLOCK_SIZE_M
    rank_beg = offs_am // m_per_rank
    rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // m_per_rank
    # token = dl.wait(barrier_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire", waitValue=1)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # a_ptrs = dl.consume_token(a_ptrs, token)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if a_dtype == tl.int8:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator.to(c_dtype), mask=c_mask)


# def cp_engine_producer_all_gather_full_mesh_pull(
#     rank,
#     num_ranks,
#     local_tensor: torch.Tensor,
#     remote_tensor_buffers: List[torch.Tensor],
#     barrier_buffers: List[torch.Tensor],
#     stream: torch.cuda.Stream,
#     for_correctness=False,
# ):
#     M_per_rank, N = local_tensor.shape

#     rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

#     with torch.xpu.stream(stream):
#         # if for_correctness:
#         #     # fake a slow communication case
#         #     # test if the computation is waiting for the correct communication
#         #     _add_noise_workload_debug()
#         for src_rank in rank_orders:
#             if src_rank == rank:
#                 continue
#             dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
#             src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
#             dst.copy_(src)

#             (err, ) = cuda.cuStreamWriteValue32(
#                 stream.cuda_stream,
#                 barrier_buffers[rank][src_rank].data_ptr(),
#                 1,
#                 cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
#             )
#             CUDA_CHECK(err)


def cp_engine_producer_all_gather_ring_push_1d(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    for_correctness=False,
):
    flag_dtype = barrier_buffers[0].dtype
    assert flag_dtype in [torch.uint32], flag_dtype
    # if flag_dtype == torch.int32:
    #     wait_value_fn = cuda.cuStreamWaitValue32
    #     write_value_fn = cuda.cuStreamWriteValue32
    # else:
    #     wait_value_fn = cuda.cuStreamWaitValue64
    #     write_value_fn = cuda.cuStreamWriteValue64

    def wait_ready(rank: int, segment: int, stream: torch.cuda.Stream):
        # (err, ) = wait_value_fn(
        #     stream.cuda_stream,
        #     barrier_buffers[rank][segment].data_ptr(),
        #     1,
        #     cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        # )
        # CUDA_CHECK(err)
        pass

    def set_ready(rank, segment, stream: torch.cuda.Stream):
        # (err, ) = write_value_fn(
        #     stream.cuda_stream,
        #     barrier_buffers[rank][segment].data_ptr(),
        #     1,
        #     cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        # )
        # CUDA_CHECK(err)
        pass

    M_per_rank, N = local_tensor.shape
    to_rank = (rank - 1 + num_ranks) % num_ranks
    with torch.xpu.stream(stream):
        # if for_correctness:
        #     # fake a slow communication case
        #     # test if the computation is waiting for the correct communication
        #     _add_noise_workload_debug()

        for stage in range(num_ranks - 1):
            send_segment = (rank + stage) % num_ranks
            M_start = send_segment * M_per_rank
            M_end = M_start + M_per_rank
            if stage != 0:
                wait_ready(rank, send_segment, stream)
            dst = remote_tensor_buffers[to_rank][M_start:M_end, :]
            src = remote_tensor_buffers[rank][M_start:M_end, :]
            dst.copy_(src)
            set_ready(to_rank, send_segment, stream)


def cp_engine_producer_all_gather_ring_push_numa_2d(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    for_correctness=False,
):
    flag_dtype = barrier_buffers[0].dtype
    # assert flag_dtype in [torch.int32, NVSHMEM_SIGNAL_DTYPE]flag_dtype
    assert flag_dtype in [torch.uint32], flag_dtype
    # if flag_dtype == torch.int32:
    #     wait_value_fn = cuda.cuStreamWaitValue32
    #     write_value_fn = cuda.cuStreamWriteValue32
    # else:
    #     wait_value_fn = cuda.cuStreamWaitValue64
    #     write_value_fn = cuda.cuStreamWriteValue64

    def wait_ready(rank: int, segment: int, stream: torch.cuda.Stream):
        # (err, ) = wait_value_fn(
        #     stream.cuda_stream,
        #     barrier_buffers[rank][segment].data_ptr(),
        #     1,
        #     cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        # )
        # CUDA_CHECK(err)
        pass

    def set_ready(rank, segment, stream: torch.cuda.Stream):
        # (err, ) = write_value_fn(
        #     stream.cuda_stream,
        #     barrier_buffers[rank][segment].data_ptr(),
        #     1,
        #     cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        # )
        # CUDA_CHECK(err)
        pass

    NUMA_WORLD_SIZE = get_numa_world_size()
    assert (num_ranks % NUMA_WORLD_SIZE == 0), f"num_ranks {num_ranks} should be divisible by NUMA {NUMA_WORLD_SIZE}"
    n_numa_nodes = num_ranks // NUMA_WORLD_SIZE
    assert n_numa_nodes == 2, f"n_numa_nodes {n_numa_nodes} should be 2"

    M_per_rank, N = local_tensor.shape
    to_rank = (rank - 1 + num_ranks) % num_ranks
    numa_node_id = rank // NUMA_WORLD_SIZE
    to_rank_numa = (rank - 1 + NUMA_WORLD_SIZE) % NUMA_WORLD_SIZE + numa_node_id * NUMA_WORLD_SIZE
    with torch.cuda.stream(stream):
        # if for_correctness:
        #     # fake a slow communication case
        #     # test if the computation is waiting for the correct communication
        #     _add_noise_workload_debug()

        for stage in range(num_ranks - 1):
            send_segment = (rank + stage) % num_ranks
            is_2d_stage = stage >= NUMA_WORLD_SIZE and rank % NUMA_WORLD_SIZE == 0
            if is_2d_stage:
                send_segment = (send_segment + NUMA_WORLD_SIZE) % num_ranks
                to_rank = to_rank_numa
            M_start = send_segment * M_per_rank
            M_end = M_start + M_per_rank
            if stage != 0:
                wait_ready(rank, send_segment, stream)
            dst = remote_tensor_buffers[to_rank][M_start:M_end, :]
            src = remote_tensor_buffers[rank][M_start:M_end, :]
            dst.copy_(src)
            set_ready(to_rank, send_segment, stream)


def cp_engine_producer_all_gather_intra_node(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    stream: torch.cuda.Stream,
    for_correctness=False,
    all_gather_method: AllGatherMethod = AllGatherMethod.All2All_IntraNode,
):
    # if all_gather_method == AllGatherMethod.All2All_IntraNode:
    #     fn = cp_engine_producer_all_gather_full_mesh_pull
    if all_gather_method == AllGatherMethod.Ring1D_IntraNode:
        fn = cp_engine_producer_all_gather_ring_push_1d
    elif all_gather_method == AllGatherMethod.Ring2D_IntraNode:
        fn = cp_engine_producer_all_gather_ring_push_numa_2d
    # else:
    #     raise Exception(f"Unsupported allgather method: {all_gather_method}")

    fn(
        rank,
        num_ranks,
        local_tensor,
        remote_tensor_buffers,
        barrier_buffers,
        stream,
        for_correctness=for_correctness,
    )


def rowise_ag_gemm_dispatcher(a, b, c, ctx: AllGatherGEMMTensorParallelContext, persistent=False, autotune=False,
                              straggler_option=None):
    current_stream = torch.xpu.current_stream()
    # if ctx.is_multinode:
    #     ctx.ag_internode_stream.wait_stream(current_stream)
    ctx.ag_intranode_stream.wait_stream(current_stream)

    # if not ctx.is_multinode:
    cp_engine_producer_all_gather_intra_node(
        ctx.rank,
        ctx.num_ranks,
        a,
        ctx.symm_workspaces,
        ctx.symm_barriers,
        ctx.ag_intranode_stream,
        for_correctness=ctx.for_correctness,
        all_gather_method=ctx.all_gather_method,
    )
    # else:
    #     cp_engine_producer_all_gather_inter_node(a, ctx.symm_workspaces, ctx.symm_barriers, ctx.barrier_target,
    #                                              ctx.rank, ctx.num_local_ranks, ctx.num_ranks, ctx.ag_intranode_stream,
    #                                              ctx.ag_internode_stream, for_correctness=ctx.for_correctness,
    #                                              all_gather_method=ctx.all_gather_method)

    if straggler_option and ctx.rank == straggler_option[0]:
        torch.xpu._sleep(straggler_option[1])

    M_per_rank, K = a.shape
    M = M_per_rank * ctx.num_ranks
    # if not persistent:
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(ctx.N_per_rank, META["BLOCK_SIZE_N"]), )
    if not autotune:
        compiled = kernel_consumer_gemm_non_persistent[grid](
            ctx.symm_workspace[:M], b, c,  #
            M, ctx.N_per_rank, ctx.K,  #
            ctx.symm_workspace.stride(0), ctx.symm_workspace.stride(1), b.stride(1), b.stride(0), c.stride(0),
            c.stride(1), ctx.rank, ctx.num_ranks, ctx.symm_barrier, ctx.BLOCK_M, ctx.BLOCK_N, ctx.BLOCK_K,
            ctx.GROUP_SIZE_M, num_stages=ctx.stages, num_warps=ctx.warps)
    # else:
    #     compiled = kernel_consumer_gemm_non_persistent_autotune[grid](
    #         ctx.symm_workspace[:M], b, c,  #
    #         M, ctx.N_per_rank, ctx.K,  #
    #         ctx.symm_workspace.stride(0), ctx.symm_workspace.stride(1), b.stride(1), b.stride(0), c.stride(0),
    #         c.stride(1), ctx.rank, ctx.num_ranks, ctx.symm_barrier)
    # else:
    #     # TMA descriptors require a global memory allocation
    #     def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    #         return torch.empty(size, device="xpu", dtype=torch.int8)

    #     triton.set_allocator(alloc_fn)

    #     internode_ag_sm = ctx.n_nodes - 1
    #     gemm_sm = ctx.max_gemm_sm - internode_ag_sm
    #     grid = lambda META: (min(
    #         gemm_sm,
    #         triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(ctx.N_per_rank, META["BLOCK_SIZE_N"]),
    #     ), )

    #     if not autotune:
    #         compiled = kernel_consumer_gemm_persistent[grid](ctx.symm_workspace[:M], b, c, M, ctx.N_per_rank, ctx.K,
    #                                                          ctx.rank, ctx.num_ranks, ctx.symm_barrier, ctx.BLOCK_M,
    #                                                          ctx.BLOCK_N, ctx.BLOCK_K, ctx.GROUP_SIZE_M, False, gemm_sm,
    #                                                          ready_value=ctx.barrier_target,
    #                                                          LOCAL_WORLD_SIZE=ctx.num_local_ranks,
    #                                                          num_stages=ctx.stages, num_warps=ctx.warps)
    #     else:
    #         compiled = kernel_consumer_gemm_persistent_autotune[grid](ctx.symm_workspace[:M], b, c, M, ctx.N_per_rank,
    #                                                                   ctx.K, ctx.rank, ctx.num_ranks, ctx.symm_barrier,
    #                                                                   LOCAL_WORLD_SIZE=ctx.num_local_ranks,
    #                                                                   EPILOGUE_SUBTILE=False, NUM_SMS=gemm_sm)

    # if ctx.is_multinode:
    #     current_stream.wait_stream(ctx.ag_internode_stream)
    current_stream.wait_stream(ctx.ag_intranode_stream)

    return compiled

def ag_gemm(a, b, ctx: AllGatherGEMMTensorParallelContext, persistent=True, autotune=False, straggler_option=None,
            use_cooperative=False): # disable cooperative grid since intel triton doesn't support it
    """allgather gemm
    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [N_per_rank, K]
        ctx: (AllGatherGEMMTensorParallelContext, Optional): if not provided, created immediately
        rank (int, Optional): current rank, used for creating AllGatherGEMMTensorParallelContext
        num_ranks (int, Optional): total number of ranks, used for creating AllGatherGEMMTensorParallelContext
        persistent (bool, Optional): whether to use persistent GEMM kernel
        autotune(bool, Optional): whether to use autotuned GEMM kernel
        straggler_option(tuple[int, int], Optional): [straggler id, straggler_latency (ns)] options for debugging straggler

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M, N_per_rank]
    """

    assert a.shape[1] == b.shape[
        1], f"tensor_B should has shape (col_major) [{b.shape[0]}, {a.shape[1]}], but get [{b.shape}]"
    assert a.dtype == b.dtype, f"Dtype of input and weight must be same: tensor_A dtype {a.dtype}, tensor_B dtype {b.dtype}"

    M_per_rank, K = a.shape
    N_per_rank, _ = b.shape

    assert a.shape[0] * ctx.num_ranks <= ctx.max_M and a.shape[
        1] == ctx.K, f"Shape of tensor_A must not exceed the maxmize M of ctx: tensor_A shape [{a.shape}], ctx shape [{ctx.max_M},{ctx.K}]"
    assert b.shape[
        0] == ctx.N_per_rank, f"N_per_rank of tensor_B must match that of ctx: tensor_B shape [{b.shape[0]}], ctx shape [{ctx.N_per_rank}]"
    assert ctx.tensor_dtype == a.dtype, f"dtype of ctx must match that of ctx: tensor_A dtype {a.dtype}, ctx dtype {ctx.tensor_dtype}"

    C = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank], dtype=a.dtype, device=a.device)

    local_copy_and_barrier_all(ctx.local_rank, ctx.rank, ctx.num_ranks, a, ctx.symm_workspace, ctx.symm_comm_buf,
                               ctx.symm_barrier, M_per_rank, K, ctx.phase, is_internode=ctx.is_multinode,
                               use_cooperative=use_cooperative)
    ctx.phase += 2

    rowise_ag_gemm_dispatcher(a, b, C, ctx, persistent=persistent, autotune=autotune, straggler_option=straggler_option)

    return C
# ===============================================================================================================

@register_test("correctness")
def test_ag_gemm(args, autotune=False):
    device = "xpu"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    M = 4091 * num_ranks
    N = 5120
    K = 1024

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    debug = args.debug
    ctx = create_ag_gemm_context(A, B, rank, num_ranks, num_local_ranks=args.local_world_size, max_M=M,
                                 for_correctness=debug)
    if rank == 0:
        print(f"all gather with: {ctx.all_gather_method}")

    def func():
        return ag_gemm(A, B, ctx=ctx, persistent=args.persistent, autotune=autotune)

    if autotune:
        _func = func
        func = contextual_autotune(is_dist=True)(lambda: _func())

    if rank == 0 and debug:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        func()
        os.environ["TRITON_ALWAYS_COMPILE"] = "0"
        os.environ["MLIR_ENABLE_DUMP"] = "0"

    # with group_profile("ag_gemm_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=args.default_group):
    for i in range(5):
        # every time, use a new input data to check correctness
        A.random_()
        B.random_()
        ctx.symm_workspace[:M].random_()
        C = func()

    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=args.default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    for i in range(num_ranks):
        torch.distributed.barrier(args.default_group)
        if rank == i:
            print(f"Rank {rank}")
            # (C_golden, C, atol=1e-3, rtol=1e-3)



register_test("correctness_autotune")(lambda args: test_ag_gemm(args, autotune=True))

configs = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}


if __name__ == "__main__":
    args = get_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.xpu.set_device(LOCAL_RANK)
    args.default_group = initialize_distributed()

    # if torch.cuda.get_device_capability() < (9, 0):
    #     if args.persistent:
    #         print("Persistent is not supported on device with capability < (9, 0). exit...")
    #         sys.exit()

    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    # nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
