# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Type stubs for the _pyishmem native module.
"""

import numpy as np

# Constants
ISHMEM_TEAM_INVALID: int
ISHMEM_TEAM_WORLD: int
ISHMEM_TEAM_SHARED: int
ISHMEM_CMP_EQ: int
ISHMEM_CMP_NE: int
ISHMEM_CMP_GT: int
ISHMEM_CMP_GE: int
ISHMEM_CMP_LT: int
ISHMEM_CMP_LE: int
ISHMEM_SIGNAL_SET: int
ISHMEM_SIGNAL_ADD: int


def ishmem_init() -> None:
    """Initialize the Intel SHMEM library."""
    ...


def ishmem_finalize() -> None:
    """Finalize the Intel SHMEM library."""
    ...


def ishmem_my_pe() -> int:
    """Return the PE number of the calling process."""
    ...


def ishmem_n_pes() -> int:
    """Return the total number of PEs."""
    ...


def ishmem_malloc(size: int) -> np.intp:
    """
    Allocate symmetric memory from the Intel SHMEM heap.

    Args:
        size: Number of bytes to allocate

    Returns:
        Pointer to the allocated memory as an integer

    Raises:
        RuntimeError: If allocation fails
    """
    ...


def ishmem_free(ptr: np.intp) -> None:
    """
    Free symmetric memory previously allocated by ishmem_malloc.

    Args:
        ptr: Pointer to memory to free (as integer)
    """
    ...


def ishmem_calloc(count: int, size: int) -> np.intp:
    """
    Allocate zero-initialized symmetric memory.

    Args:
        count: Number of elements to allocate
        size: Size of each element in bytes

    Returns:
        Pointer to the allocated memory as an integer
    """
    ...


def ishmem_align(alignment: int, size: int) -> np.intp:
    """
    Allocate aligned symmetric memory.

    Args:
        alignment: Alignment requirement in bytes
        size: Number of bytes to allocate

    Returns:
        Pointer to the allocated memory as an integer
    """
    ...


def ishmem_ptr(dest: np.intp, pe: int) -> np.intp:
    """
    Get a pointer to a symmetric data object on a specified PE.

    Args:
        dest: Pointer to symmetric data object
        pe: Target PE number

    Returns:
        Pointer to the data on the target PE
    """
    ...


def ishmem_barrier_all() -> None:
    """Barrier synchronization across all PEs."""
    ...


def ishmem_sync_all() -> None:
    """Synchronize all PEs."""
    ...


def ishmem_fence() -> None:
    """Ensure ordering of operations."""
    ...


def ishmem_quiet() -> None:
    """Wait for completion of all outstanding operations."""
    ...

