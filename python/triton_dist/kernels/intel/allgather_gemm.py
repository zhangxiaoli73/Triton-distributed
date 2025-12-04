import logging
import os

import torch
import triton
import triton.language as tl

from typing import Optional, List
from dataclasses import dataclass, field

from triton_dist.kernels.intel.common_ops import barrier_all_intra_node_non_atomic, wait_signal_range
from triton_dist.kernels.intel.allgather import AllGatherMethod, cp_engine_producer_all_gather_intra_node, get_auto_all_gather_method, cp_engine_producer_all_gather_inter_node
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, nvshmem_barrier_all_on_stream
from triton_dist.kernels.intel.symm_utils import ishmem_create_tensors

# Setup logging for debugging
_log_level = os.environ.get("TRITON_DIST_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.WARNING),
                    format='[%(asctime)s][Rank %(rank)s][%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
_logger = logging.getLogger(__name__)


class RankLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that includes rank information."""
    def process(self, msg, kwargs):
        kwargs.setdefault('extra', {})['rank'] = self.extra.get('rank', '?')
        return msg, kwargs


def get_logger(rank: int = -1):
    """Get a logger with rank information."""
    return RankLoggerAdapter(_logger, {'rank': rank})


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
    use_cooperative: tl.constexpr = False,
):
    """
    Intel SYCL implementation of copy + barrier kernel.

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

    # Set symm barrier signal - only first work-group does this
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


def local_copy_and_barrier_all(local_rank, rank, num_ranks, local_data, global_data, comm_buf, barrier_ptr, M_per_rank,
                               N, phase, is_internode: bool = False, use_cooperative: bool = False):
    """
    Intel XPU implementation of local copy with barrier synchronization.

    Args:
        local_rank: Local rank within the node
        rank: Global rank
        num_ranks: Total number of ranks
        local_data: Source tensor to copy
        global_data: Destination tensor (symmetric workspace)
        comm_buf: IPC memory pointers for barrier synchronization
        barrier_ptr: Barrier signal buffer
        M_per_rank: Rows per rank
        N: Number of columns
        phase: Synchronization phase counter
        is_internode: Whether this is inter-node communication
        use_cooperative: Ignored on Intel (cooperative launch not supported)
    """
    if not is_internode:
        # intra node
        grid = lambda META: (min(
            triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            torch.xpu.get_device_properties("xpu").gpu_eu_count), )
        additional_options = {}
        CPY_BLOCKS = [128, 256]

        copy_and_barrier_all_intra_node_kernel[grid](local_rank, rank, num_ranks, local_data,
                                                     global_data, barrier_ptr, comm_buf, M_per_rank, N,
                                                     local_data.stride(0), local_data.stride(1), global_data.stride(0),
                                                     global_data.stride(1), phase, CPY_BLOCKS[0], CPY_BLOCKS[1],
                                                     use_cooperative, **additional_options)
    else:
        raise ValueError(f"inter-nodes cannot support now on XPU")


@triton.jit
def swizzle_2d(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# TMA related test
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

def _kernel_consumer_gemm_non_persistent_repr(proxy):
    """Generate a unique kernel name for Intel XPU."""
    constexprs = proxy.constants
    # Intel XPU: use a generic device identifier
    device_name = "xpu"
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]
    if constexprs.get("stride_am", None) == 1:  # column major => n
        a_trans = "n"
    elif constexprs.get("stride_ak", None) == 1:  # row-major => t
        a_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_bk", None) == 1:
        b_trans = "n"
    elif constexprs.get("stride_bn", None) == 1:
        b_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    if constexprs.get("stride_cm", None) == 1:
        c_trans = "n"
    elif constexprs.get("stride_cn", None) == 1:
        c_trans = "t"
    else:
        raise Exception("both stride_am/stride_ak != 1")

    return f"triton3x_{device_name}_ag_gemm_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_{a_trans}{b_trans}{c_trans}"


@triton.jit(do_not_specialize=["rank"], launch_metadata=_matmul_launch_metadata,
            repr=_kernel_consumer_gemm_non_persistent_repr)
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
    # Intel XPU: use polling-based wait instead of dl.wait
    wait_signal_range(barrier_ptr, rank_beg, rank_end - rank_beg + 1, 1)

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

    # Note: dl.consume_token is not needed on Intel XPU

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


def matmul_get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8}, num_stages=s,
                      num_warps=w)
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]

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
    symm_workspaces: List[torch.Tensor] = field(init=False)  # [max_M, K] ag buffer to copy remote data to local symm ws
    symm_barriers: List[torch.Tensor] = field(init=False)  # signal buffer to sync
    symm_workspace: torch.Tensor = field(init=False)
    symm_barrier: torch.Tensor = field(init=False)
    fake_barrier: torch.Tensor = field(init=False)  # for gemm only function
    symm_comm_buf: torch.Tensor = field(init=False)
    barrier_target = 1
    # async streams (Intel XPU streams)
    ag_intranode_stream: Optional[torch.xpu.Stream] = None
    ag_internode_stream: Optional[torch.xpu.Stream] = None
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
        log = get_logger(self.rank)
        log.info(f"[__post_init__] Starting initialization: num_ranks={self.num_ranks}, num_local_ranks={self.num_local_ranks}")

        assert self.num_ranks % self.num_local_ranks == 0
        self.is_multinode = self.num_ranks > self.num_local_ranks
        self.n_nodes = self.num_ranks // self.num_local_ranks
        self.node_rank = self.rank // self.num_local_ranks
        self.local_rank = self.rank % self.num_local_ranks
        log.info(f"[__post_init__] is_multinode={self.is_multinode}, local_rank={self.local_rank}")

        # create symmetric workspace from pytorch symmetric memory
        log.info(f"[__post_init__] Creating symm_workspaces: shape=({self.max_M}, {self.K}), dtype={self.tensor_dtype}")
        self.symm_workspaces = ishmem_create_tensors((self.max_M, self.K), self.tensor_dtype, self.rank,
                                                      self.num_local_ranks)
        self.symm_workspace = self.symm_workspaces[self.local_rank]
        log.info(f"[__post_init__] symm_workspaces created successfully, got {len(self.symm_workspaces)} tensors")

        log.info(f"[__post_init__] Creating symm_comm_buf: shape=({3 * self.num_ranks},)")
        self.symm_comm_buf = ishmem_create_tensors((3 * self.num_ranks, ), torch.int32, self.rank, self.num_local_ranks)
        self.symm_comm_buf = self.symm_comm_buf[self.local_rank]
        self.symm_comm_buf.fill_(0)
        log.info(f"[__post_init__] symm_comm_buf created successfully")

        barrier_dtype = NVSHMEM_SIGNAL_DTYPE if self.is_multinode else torch.int32
        log.info(f"[__post_init__] Creating symm_barriers: shape=({self.num_ranks},), dtype={barrier_dtype}")
        self.symm_barriers = ishmem_create_tensors((self.num_ranks, ), barrier_dtype, self.rank, self.num_local_ranks)
        self.symm_barrier = self.symm_barriers[self.local_rank]
        self.symm_barrier.fill_(0)
        log.info(f"[__post_init__] symm_barriers created successfully")

        self.fake_barrier = torch.ones([self.num_ranks], dtype=barrier_dtype, device="xpu")
        self.max_gemm_sm = torch.xpu.get_device_properties("xpu").gpu_eu_count

        torch.xpu.synchronize()
        log.info(f"[__post_init__] Initialization complete")

    def finailize(self):
        # Note: On Intel XPU, we use IPC memory which is managed differently
        # The tensors will be freed when they go out of scope
        pass


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


def ag_gemm(a, b, ctx: AllGatherGEMMTensorParallelContext, persistent=True, autotune=False, straggler_option=None):
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
                               ctx.symm_barrier, M_per_rank, K, ctx.phase, is_internode=ctx.is_multinode)
    ctx.phase += 2

    rowise_ag_gemm_dispatcher(a, b, C, ctx, persistent=persistent, autotune=autotune, straggler_option=straggler_option)

    return C


def rowise_ag_gemm_dispatcher(a, b, c, ctx: AllGatherGEMMTensorParallelContext, persistent=False, autotune=False,
                              straggler_option=None):
    log = get_logger(ctx.rank)
    log.info(f"[rowise_ag_gemm_dispatcher] Starting: a.shape={a.shape}, b.shape={b.shape}, persistent={persistent}")

    current_stream = torch.xpu.current_stream()
    if ctx.is_multinode:
        ctx.ag_internode_stream.wait_stream(current_stream)
    ctx.ag_intranode_stream.wait_stream(current_stream)
    log.info(f"[rowise_ag_gemm_dispatcher] Streams synchronized, starting AllGather")

    if not ctx.is_multinode:
        log.info(f"[rowise_ag_gemm_dispatcher] Calling cp_engine_producer_all_gather_intra_node with method={ctx.all_gather_method}")
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
        log.info(f"[rowise_ag_gemm_dispatcher] AllGather completed")
    else:
        raise ValueError(f"Inter-nodes not supported on XPU")

    M_per_rank, _ = a.shape
    M = M_per_rank * ctx.num_ranks
    log.info(f"[rowise_ag_gemm_dispatcher] Starting GEMM: M={M}, N={ctx.N_per_rank}, K={ctx.K}")

    if not persistent:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(ctx.N_per_rank, META["BLOCK_SIZE_N"]), )
        if not autotune:
            log.info(f"[rowise_ag_gemm_dispatcher] Launching kernel_consumer_gemm_non_persistent")
            compiled = kernel_consumer_gemm_non_persistent[grid](
                ctx.symm_workspace[:M], b, c,  #
                M, ctx.N_per_rank, ctx.K,  #
                ctx.symm_workspace.stride(0), ctx.symm_workspace.stride(1), b.stride(1), b.stride(0), c.stride(0),
                c.stride(1), ctx.rank, ctx.num_ranks, ctx.symm_barrier, ctx.BLOCK_M, ctx.BLOCK_N, ctx.BLOCK_K,
                ctx.GROUP_SIZE_M, num_stages=ctx.stages, num_warps=ctx.warps)
            log.info(f"[rowise_ag_gemm_dispatcher] GEMM kernel launched")
        else:
            raise ValueError(f"Autotune not supported on XPU")
    else:
        raise ValueError(f"Persistent not supported on XPU")

    log.info(f"[rowise_ag_gemm_dispatcher] Waiting for intranode stream")
    current_stream.wait_stream(ctx.ag_intranode_stream)
    log.info(f"[rowise_ag_gemm_dispatcher] Completed")

    return compiled


def gemm_persistent(a, b, ctx: AllGatherGEMMTensorParallelContext, autotune=False):
    """
    Persistent GEMM kernel - NOT SUPPORTED on Intel XPU.

    Intel XPU does not support persistent kernels with TMA descriptors.
    Use gemm_non_persistent instead.
    """
    raise NotImplementedError("Persistent GEMM is not supported on Intel XPU. Use gemm_non_persistent instead.")


def gemm_non_persistent(a, b, ctx: AllGatherGEMMTensorParallelContext):
    """
    Non-persistent GEMM kernel for Intel XPU.
    """
    M, K = a.shape
    N, _ = b.shape
    C = torch.empty([M, N], dtype=a.dtype, device=a.device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel_consumer_gemm_non_persistent[grid](
        a,
        b,
        C,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        C.stride(0),
        C.stride(1),
        ctx.rank,
        ctx.num_ranks,
        ctx.fake_barrier,
        ctx.BLOCK_M,
        ctx.BLOCK_N,
        ctx.BLOCK_K,
        8,
        num_stages=ctx.stages,
        num_warps=8,
    )

    return C
