import datetime
import functools
import gzip
import json
import logging
import os
import random
import re
import shutil
import string
import subprocess
import sys
from contextlib import contextmanager, nullcontext, redirect_stdout
from multiprocessing import Pool, cpu_count
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from functools import wraps

import numpy as np
import packaging.version
import torch

from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch._C._distributed_c10d import _SymmetricMemory

# Some code from python/flux/util.py in flux project

_TP_GROUP = None
_group_name_to_workspace_tensor: dict[str, torch.Tensor | None] = {}

def init_seed(seed=0):
    torch.use_deterministic_algorithms(True, warn_only=True)
    # zero empty takes more kernel launch and may hide uninitialized problem. always set to False
    # available since torch 2.2: https://docs.pytorch.org/docs/2.2/deterministic.html
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:
        logging.warning("torch.utils.fill_uninitialized_memory is available only for torch >=2.2")
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    np.random.seed(3 + seed)
    random.seed(3 + seed)

def ishmem_create_tensor(shape, dtype) -> torch.Tensor:
    torch.xpu.synchronize()
    tensor = torch.tensor(shape, dtype=dtype)
    torch.xpu.synchronize()
    return tensor

def create_symmetric_handle(shape, dtype, group_name) -> _SymmetricMemory:
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
        return local_sm_handle.get_buffer(peer, tuple(t.size()), t.dtype)

    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    rank_on_same_node_end = rank_on_same_node_start + local_world_size
    return [_get_peer_tensor(local_tensor, peer) for peer in range(rank_on_same_node_start, rank_on_same_node_end)]

def ishmem_create_tensors(shape, dtype, rank, local_world_size) -> List[torch.Tensor]:
    """
    Create IPC-shared tensors for Intel XPU.

    Note: This is a placeholder implementation. Full IPC memory sharing
    on Intel XPU requires platform-specific implementation using Level Zero
    or similar APIs.

    For now, this creates local tensors. Proper IPC sharing needs to be
    implemented based on the Intel XPU runtime capabilities.
    """
    local_rank = rank % local_world_size
    torch.xpu.synchronize()

    # Create a list of tensors (placeholder - each rank creates its own), WA with pytorch symmetric memory

    gp = torch.distributed.new_group(ranks=list(range(local_world_size)), backend="xccl")
    symm_handle = create_symmetric_handle(shape, dtype, gp.group_name)
    symm_comm = get_remote_tensors(symm_handle, _group_name_to_workspace_tensor[symm_handle], rank, local_world_size)
    return symm_comm


def finalize_distributed():
    torch.distributed.destroy_process_group()


class TorchStreamWrapper:
    """Wrapper for Intel XPU stream."""

    def __init__(self, pt_stream: torch.xpu.Stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.sycl_queue if hasattr(pt_stream, 'sycl_queue') else None

    def __xpu_stream__(self):
        # Return stream handle for Intel XPU
        if hasattr(self.pt_stream, 'sycl_queue'):
            return self.pt_stream.sycl_queue
        return None


def ishmem_signal_wait(signal: torch.Tensor, pe: int, signal_val: int, signal_op: int,
                       stream: Optional[torch.xpu.Stream] = None) -> None:
    """
    Wait for a signal on Intel XPU.

    Note: This is a placeholder using polling. Intel XPU doesn't have
    native NVSHMEM-like signal wait operations.
    """
    # Simple polling-based wait
    while signal.item() != signal_val:
        pass


def initialize_distributed(seed=None) -> torch.distributed.ProcessGroup:
    """
    Initialize distributed environment for Intel XPU.
    """
    global _TP_GROUP
    assert _TP_GROUP is None, "TP_GROUP has already been initialized"

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    # Set Intel XPU device
    torch.xpu.set_device(LOCAL_RANK)

    torch.distributed.init_process_group(
        backend="cpu:gloo,xpu:xccl",  # Use CCL for Intel XPU
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()

    # Use CCL backend for Intel XPU tensor parallelism
    _TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="xccl")
    torch.distributed.barrier(_TP_GROUP)

    _TP_GROUP_GLOO = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")
    torch.distributed.barrier(_TP_GROUP_GLOO)

    init_seed(seed=seed if seed is not None else RANK)
    # Note: NVSHMEM is not available on Intel XPU, IPC sharing handled differently
    return _TP_GROUP


@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return False


def _make_tensor(
    shape: List[Union[int, Callable[[], int]]],
    dtype: torch.dtype,
    init_args: Union[Tuple[float, float], Tuple[int, int]],
    device: str = "xpu",
):
    """
    rand() * scale + bias
    randint(-scale, scale) + bias
    """
    if isinstance(shape, Sequence):
        shape = tuple([x() if isinstance(x, Callable) else x for x in shape])
    elif isinstance(shape, int):
        shape = (shape, )
    elif isinstance(shape, Callable):
        shape = shape()
    else:
        raise ValueError(f"unsupported shape {shape}")

    scale, bias = init_args
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        out = (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * scale + bias
    elif dtype == torch.int8:
        out = torch.randint(-scale, scale, shape, dtype=torch.int8, device=device)
        out = out + bias
    elif is_fp8_dtype(dtype):
        out = (torch.rand(shape, dtype=torch.float16, device=device) * 2 - 1) * scale + bias
        with with_torch_deterministic(False):
            out = out.to(dtype)
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    return out


def generate_data(configs):
    while True:
        yield (_make_tensor(*args) if args else None for args in configs)


def get_torch_prof_ctx(do_prof: bool):
    """
    Get profiler context for Intel XPU.

    Note: Intel XPU profiling may require different setup.
    Using CPU profiling as fallback.
    """
    if do_prof:
        activities = [torch.profiler.ProfilerActivity.CPU]
        # Try to add XPU activity if available
        if hasattr(torch.profiler.ProfilerActivity, 'XPU'):
            activities.append(torch.profiler.ProfilerActivity.XPU)
        ctx = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
        )
    else:
        ctx = nullcontext()
    return ctx
