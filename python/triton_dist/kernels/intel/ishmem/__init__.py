# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Intel SHMEM integration for Triton Distributed.

This module provides Python APIs for Intel SHMEM (ishmem) memory management,
enabling device-initiated shared memory communication on Intel GPUs.

Key functions:
    - ishmem_malloc: Allocate symmetric memory from the Intel SHMEM heap
    - ishmem_free: Free previously allocated symmetric memory
    - ishmem_init: Initialize the Intel SHMEM library
    - ishmem_finalize: Finalize the Intel SHMEM library

Example:
    from triton_dist.kernels.intel.ishmem import pyishmem
    
    # Initialize
    pyishmem.ishmem_init()
    
    # Allocate 1MB of symmetric memory
    ptr = pyishmem.ishmem_malloc(1024 * 1024)
    
    # Use the memory with Triton kernels...
    
    # Free and finalize
    pyishmem.ishmem_free(ptr)
    pyishmem.ishmem_finalize()
"""

# Re-export pyishmem module when built
try:
    from . import pyishmem
except ImportError:
    # pyishmem may not be built yet
    pyishmem = None

