# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Python bindings for Intel SHMEM (ishmem) library.

This module provides Python APIs for Intel SHMEM memory management functions,
including ishmem_malloc and ishmem_free for symmetric memory allocation.

Usage:
    import pyishmem
    
    # Initialize ishmem
    pyishmem.ishmem_init()
    
    # Allocate symmetric memory
    ptr = pyishmem.ishmem_malloc(1024)  # Allocate 1024 bytes
    
    # Use the memory...
    
    # Free the memory
    pyishmem.ishmem_free(ptr)
    
    # Finalize
    pyishmem.ishmem_finalize()
"""

import os
import sys
from typing import Optional

# Try to import the native module
try:
    from _pyishmem import (
        ishmem_init,
        ishmem_finalize,
        ishmem_my_pe,
        ishmem_n_pes,
        ishmem_malloc,
        ishmem_free,
        ishmem_calloc,
        ishmem_align,
        ishmem_ptr,
        ishmem_barrier_all,
        ishmem_sync_all,
        ishmem_fence,
        ishmem_quiet,
        ISHMEM_TEAM_INVALID,
        ISHMEM_TEAM_WORLD,
        ISHMEM_TEAM_SHARED,
        ISHMEM_CMP_EQ,
        ISHMEM_CMP_NE,
        ISHMEM_CMP_GT,
        ISHMEM_CMP_GE,
        ISHMEM_CMP_LT,
        ISHMEM_CMP_LE,
        ISHMEM_SIGNAL_SET,
        ISHMEM_SIGNAL_ADD,
    )
    from _pyishmem import *  # noqa: F403
except ImportError as e:
    print(
        "Failed to import _pyishmem native module. "
        "Please ensure Intel SHMEM library is installed and ISHMEM_HOME is set correctly. "
        "Also add the ISHMEM library path to LD_LIBRARY_PATH.",
        flush=True,
        file=sys.stderr,
    )
    raise e

__all__ = [
    # Initialization
    "ishmem_init",
    "ishmem_finalize",
    # Query
    "ishmem_my_pe",
    "ishmem_n_pes",
    # Memory management
    "ishmem_malloc",
    "ishmem_free",
    "ishmem_calloc",
    "ishmem_align",
    "ishmem_ptr",
    # Synchronization
    "ishmem_barrier_all",
    "ishmem_sync_all",
    "ishmem_fence",
    "ishmem_quiet",
    # Constants
    "ISHMEM_TEAM_INVALID",
    "ISHMEM_TEAM_WORLD",
    "ISHMEM_TEAM_SHARED",
    "ISHMEM_CMP_EQ",
    "ISHMEM_CMP_NE",
    "ISHMEM_CMP_GT",
    "ISHMEM_CMP_GE",
    "ISHMEM_CMP_LT",
    "ISHMEM_CMP_LE",
    "ISHMEM_SIGNAL_SET",
    "ISHMEM_SIGNAL_ADD",
]


def get_ishmem_home() -> Optional[str]:
    """Get the Intel SHMEM installation directory from environment."""
    return os.environ.get("ISHMEM_HOME", None)


def get_ishmem_lib_path() -> Optional[str]:
    """Get the Intel SHMEM library path."""
    home = get_ishmem_home()
    if home:
        return os.path.join(home, "lib")
    return None

