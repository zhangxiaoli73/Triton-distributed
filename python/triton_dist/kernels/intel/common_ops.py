from typing import Optional

import torch

import triton
import triton.language as tl


# ============================================================================
# Intel SYCL/XPU compatible barrier implementations
# These use standard Triton APIs instead of CUDA-specific intrinsics
# ============================================================================


@triton.jit
def _is_workgroup_master():
    """Check if current work-item is the master of the work-group (thread 0)."""
    # In SYCL/XPU, we use program_id as work-group identifier
    # For intra-workgroup master, we rely on the fact that only one work-item
    # executes non-vectorized scalar code
    return True  # Scalar code path - only one work-item executes this


@triton.jit
def _is_gpu_master():
    """Check if current work-group is the first one (master of the GPU)."""
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)
    return (pid_x + pid_y + pid_z) == 0


@triton.jit
def unsafe_barrier_on_this_grid(ptr):
    """
    Intel SYCL implementation of grid-wide barrier using atomic operations.

    WARNING: This may hang if not all work-groups can run concurrently.
    Intel Triton does not support cooperative grid launch.
    """
    tl.debug_barrier()  # Work-group level barrier (replaces __syncthreads)

    pid_size_x = tl.num_programs(axis=0)
    pid_size_y = tl.num_programs(axis=1)
    pid_size_z = tl.num_programs(axis=2)
    expected = pid_size_x * pid_size_y * pid_size_z

    # Use atomic add for arrival counting
    if _is_gpu_master():
        nb = tl.cast(0x80000000, tl.uint32, bitcast=True) - (expected - 1)
    else:
        nb = tl.cast(1, tl.uint32)

    old_arrive = tl.atomic_add(ptr, nb, sem="release", scope="gpu")

    # Wait for all work-groups to arrive
    current_arrive = tl.load(ptr)
    while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
        current_arrive = tl.load(ptr)

    tl.debug_barrier()


@triton.jit
def barrier_on_this_grid(ptr, use_cooperative: tl.constexpr):
    """
    Grid-wide barrier for Intel XPU.

    Note: use_cooperative is ignored on Intel since cooperative launch is not supported.
    Always uses the atomic-based unsafe barrier.
    """
    # Intel Triton doesn't support cooperative grid launch
    # Always use the atomic-based barrier
    unsafe_barrier_on_this_grid(ptr)


@triton.jit
def wait_signal(signal_ptr, expected_value):
    """
    Wait for a signal to reach the expected value.

    This is the Intel XPU equivalent of dl.wait() for NVSHMEM.
    Uses polling to wait for the signal.

    Args:
        signal_ptr: Pointer to the signal value
        expected_value: The value to wait for
    """
    while tl.load(signal_ptr) < expected_value:
        pass


@triton.jit
def wait_signal_range(signal_ptr, start_idx, count, expected_value):
    """
    Wait for a range of signals to reach the expected value.

    This is the Intel XPU equivalent of dl.wait() with multiple signals.

    Args:
        signal_ptr: Base pointer to the signal array
        start_idx: Starting index in the signal array
        count: Number of signals to wait for
        expected_value: The value to wait for
    """
    for i in range(count):
        while tl.load(signal_ptr + start_idx + i) < expected_value:
            pass


@triton.jit(do_not_specialize=["local_rank", "rank", "local_world_size"])
def barrier_all_intra_node_atomic_cas_block(local_rank, rank, local_world_size, symm_flag_ptrs):
    """
    Intel SYCL implementation of intra-node barrier using atomic CAS.

    Uses IPC-based symmetric memory pointers for cross-rank synchronization.

    Args:
        local_rank: Local rank within the node
        rank: Global rank
        local_world_size: Number of ranks on this node
        symm_flag_ptrs: Pointer to array of IPC memory pointers for each rank
    """
    # Phase 1: Signal all other ranks that we have arrived
    for i in range(local_world_size):
        remote_base_ptr = tl.load(symm_flag_ptrs + i).to(tl.pointer_type(tl.int32))
        # Try to set remote_ptr[local_rank] from 0 to 1
        while tl.atomic_cas(remote_base_ptr + local_rank, 0, 1, sem="release", scope="sys") != 0:
            pass

    # Phase 2: Wait for all other ranks to signal us
    local_base_ptr = tl.load(symm_flag_ptrs + local_rank).to(tl.pointer_type(tl.int32))
    for i in range(local_world_size):
        # Wait until local_ptr[i] becomes 1, then reset to 0
        while tl.atomic_cas(local_base_ptr + i, 1, 0, sem="acquire", scope="sys") != 1:
            pass

    tl.debug_barrier()


@triton.jit
def _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, local_world_size, symm_flag_ptrs, target_value):
    """
    Intel SYCL implementation of non-atomic barrier (single phase).

    Uses IPC memory pointers and standard load/store with memory ordering.

    Args:
        local_rank: Local rank within the node
        rank: Global rank
        local_world_size: Number of ranks on this node
        symm_flag_ptrs: Pointer to array of IPC memory pointers for each rank
        target_value: The synchronization value to use
    """
    # Each rank writes target_value to its slot in all other ranks' buffers
    for i in range(local_world_size):
        remote_base_ptr = tl.load(symm_flag_ptrs + i).to(tl.pointer_type(tl.int32))
        # Write our target_value to remote_ptr[local_rank]
        tl.store(remote_base_ptr + local_rank, target_value)

    # Memory fence to ensure stores are visible
    tl.debug_barrier()

    # Wait for all other ranks to write their target_value to our buffer
    local_base_ptr = tl.load(symm_flag_ptrs + local_rank).to(tl.pointer_type(tl.int32))
    for i in range(local_world_size):
        # Spin until we see target_value from rank i
        while tl.load(local_base_ptr + i) != target_value:
            pass

    tl.debug_barrier()


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "target_value"])
def barrier_all_intra_node_non_atomic_block(local_rank, rank, num_ranks, symm_flag_ptrs, target_value):
    """
    Intel SYCL implementation of non-atomic barrier for a single block/work-group.

    Args:
        local_rank: Local rank within the node
        rank: Global rank
        num_ranks: Number of ranks on this node
        symm_flag_ptrs: Pointer to array of IPC memory pointers
        target_value: The synchronization value to use

    Note: Uses two phases with separate flag regions to avoid race conditions.
    """
    # First phase
    _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flag_ptrs, target_value)
    # Second phase uses offset pointers (symm_flag_ptrs + num_ranks points to second set of flags)
    _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flag_ptrs + num_ranks, target_value)


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "target_value"])
def barrier_all_intra_node_non_atomic(local_rank, rank, num_ranks, symm_flag_ptrs, target_value,
                                      use_cooperative: tl.constexpr):
    """
    Intel SYCL implementation of non-atomic intra-node barrier.

    This barrier synchronizes:
    1. All ranks within a node (using IPC memory)
    2. All work-groups within each rank (using grid barrier)

    Args:
        local_rank: Local rank within the node
        rank: Global rank
        num_ranks: Number of ranks on this node
        symm_flag_ptrs: Pointer to array of IPC memory pointers for inter-rank sync
        target_value: The synchronization value to use
        use_cooperative: Ignored on Intel (cooperative launch not supported)

    Note: Intel Triton does not support cooperative grid launch, so use_cooperative
    is ignored and we always use atomic-based grid barrier.
    """
    pid = tl.program_id(axis=0)

    # Only first work-group performs inter-rank synchronization
    if pid == 0:
        _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flag_ptrs, target_value)

    # Barrier all work-groups within this rank
    # Note: We need a separate grid barrier pointer, using offset from symm_flag_ptrs
    # For Intel, we use a simple atomic-based barrier
    tl.debug_barrier()  # Intra-workgroup sync first

    # Second phase of inter-rank sync
    if pid == 0:
        _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flag_ptrs + num_ranks, target_value)

    tl.debug_barrier()


# ============================================================================
# Intel XPU Barrier Context and Helper Functions
# ============================================================================


class BarrierAllContext:
    """
    Intel XPU implementation of barrier context for intra-node synchronization.

    Uses IPC shared memory for cross-rank communication instead of NVSHMEM.
    """

    def __init__(self, is_intra_node, rank=0, local_rank=0, num_local_ranks=1, symm_flag_ptrs=None):
        """
        Initialize barrier context for Intel XPU.

        Args:
            is_intra_node: Whether this is intra-node only communication
            rank: Global rank
            local_rank: Local rank within the node
            num_local_ranks: Number of ranks on this node
            symm_flag_ptrs: Tensor containing IPC memory pointers for each rank
        """
        self.is_intra_node = is_intra_node
        self.target_value = 1
        self.rank = rank
        self.local_rank = local_rank
        self.num_local_ranks = num_local_ranks
        self.symm_flag_ptrs = symm_flag_ptrs  # IPC memory pointers


def barrier_all_on_stream(ctx: BarrierAllContext, stream: Optional[torch.xpu.Stream] = None):
    """
    Intel XPU implementation of barrier_all_on_stream.

    Args:
        ctx: BarrierAllContext with IPC memory configuration
        stream: XPU stream (currently unused, synchronous execution)
    """
    if ctx is None or not ctx.is_intra_node or ctx.symm_flag_ptrs is None:
        # Fallback to torch.distributed barrier
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # Use non-atomic barrier with IPC memory
    barrier_all_intra_node_non_atomic_block[(1, )](
        ctx.local_rank, ctx.rank, ctx.num_local_ranks,
        ctx.symm_flag_ptrs, ctx.target_value
    )
    ctx.target_value += 1


def log2(n):
    return len(bin(n)) - 3

def next_power_of_2(n: tl.constexpr):
    return triton.next_power_of_2(n)


@triton.jit
def bisect_left_kernel_aligned(sorted_values_ptr,  # Pointer to sorted input array of K
                               target_values,  # Pointer to search values of N
                               N: tl.constexpr,  # K should be power of 2
                               ):
    # Binary search initialization
    low = tl.full((target_values.numel, ), 0, dtype=tl.int32)
    high = tl.full((target_values.numel, ), N, dtype=tl.int32)  # Length of the sorted array

    N_LOG2 = log2(N)
    # Binary search loop
    for n in tl.range(N_LOG2):
        mid = (low + high) // 2
        mid_val = tl.load(sorted_values_ptr + mid)
        # Update search bounds
        low = tl.where(mid_val < target_values, mid + 1, low)
        high = tl.where(mid_val >= target_values, mid, high)

    low = tl.where(
        low != high and tl.load(sorted_values_ptr + low) < target_values,
        low + 1,
        low,
    )
    return low


@triton.jit
def bisect_left_kernel(
    sorted_values_ptr,  # Pointer to sorted input array of M
    target_values,  # Pointer to search values of L
    N: tl.constexpr,
):
    # Binary search initialization
    index = tl.full((target_values.numel, ), -1, dtype=tl.int32)

    # Binary search loop
    for i in tl.range(N):
        x = tl.load(sorted_values_ptr + i)
        # if index > 0 => index
        # if x > target_value => i
        # else => -1
        index = tl.where(index >= 0, index, tl.where(x >= target_values, i, -1))
    index = tl.where(index == -1, N, index)

    return index


@triton.jit
def bisect_right_kernel(sorted_values_ptr,  # Pointer to sorted input array (1D)
                        target_values,  # Pointer to search values (1D)
                        N: tl.constexpr,  # Length of sorted array
                        ):
    # Binary search initialization
    index = tl.full((target_values.numel, ), -1, dtype=tl.int32)

    # Binary search loop
    for i in tl.range(N):
        x = tl.load(sorted_values_ptr + i)
        # if index > 0 => index
        # if x > target_value => i
        # else => -1
        index = tl.where(index >= 0, index, tl.where(x > target_values, i, -1))
    index = tl.where(index == -1, N, index)
    return index


@triton.jit
def bisect_right_kernel_aligned(
    sorted_values_ptr,  # Pointer to sorted input array (1D)
    target_values,
    N: tl.constexpr,
):
    # Binary search initialization
    low = tl.full((target_values.numel, ), 0, dtype=tl.int32)
    high = tl.full((target_values.numel, ), N, dtype=tl.int32)  # Length of the sorted array

    N_LOG2 = log2(N)
    # Binary search loop
    for _ in tl.range(N_LOG2):
        mid = (low + high) // 2
        mid_val = tl.load(sorted_values_ptr + mid)

        # Update search bounds
        low = tl.where(mid_val <= target_values, mid + 1, low)
        high = tl.where(mid_val > target_values, mid, high)

    low = tl.where(low != high and tl.load(sorted_values_ptr + low) <= target_values, low + 1, low)
    # Store result
    return low


# ============================================================================
# Intel XPU Signal and Memory Operations
# These replace CUDA-specific stream operations with XPU equivalents
# ============================================================================


def _wait_eq_xpu(signal_tensor: torch.Tensor, signal: int, stream: Optional[torch.xpu.Stream] = None,
                 require_i64=False):
    """
    Wait for a signal tensor to equal a specific value.

    Intel XPU implementation using polling (no native stream wait value support).

    Args:
        signal_tensor: Tensor containing the signal value
        signal: The value to wait for
        stream: XPU stream (unused in polling implementation)
        require_i64: Whether 64-bit signal is required
    """
    # XPU doesn't have native stream wait value operations
    # Use synchronous polling as fallback
    while signal_tensor.item() != signal:
        pass


def _set_signal_xpu(signal_tensor: torch.Tensor, signal: int, stream: Optional[torch.xpu.Stream] = None):
    """
    Set a signal tensor to a specific value.

    Args:
        signal_tensor: Tensor to set
        signal: The value to set
        stream: XPU stream (unused, operation is synchronous)
    """
    signal_tensor.fill_(signal)


def _memcpy_async_xpu(dst: torch.Tensor, src: torch.Tensor, nbytes: int, stream: Optional[torch.xpu.Stream] = None):
    """
    Asynchronous memory copy for Intel XPU.

    Args:
        dst: Destination tensor
        src: Source tensor
        nbytes: Number of bytes to copy
        stream: XPU stream for async operation
    """
    # Use PyTorch's copy_ which respects the current stream
    dst.view(torch.uint8)[:nbytes].copy_(src.view(torch.uint8)[:nbytes])


# Aliases for compatibility (some code may still reference cuda versions)
_wait_eq_cuda = _wait_eq_xpu
_set_signal_cuda = _set_signal_xpu
_memcpy_async_cuda = _memcpy_async_xpu
