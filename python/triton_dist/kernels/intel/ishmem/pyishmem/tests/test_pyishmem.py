# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Unit tests for pyishmem - Python bindings for Intel SHMEM.

To run these tests, Intel SHMEM must be installed and the pyishmem
module must be built. Run with:
    pytest test_pyishmem.py -v

For multi-PE tests, use mpirun:
    mpirun -n 2 pytest test_pyishmem.py -v
"""

import pytest
import sys


# Skip all tests if pyishmem is not available
try:
    import pyishmem
    PYISHMEM_AVAILABLE = True
except ImportError:
    PYISHMEM_AVAILABLE = False
    pyishmem = None


pytestmark = pytest.mark.skipif(
    not PYISHMEM_AVAILABLE,
    reason="pyishmem module not available"
)


class TestIshmemInit:
    """Tests for ishmem initialization and finalization."""

    def test_init_finalize(self):
        """Test basic init and finalize."""
        pyishmem.ishmem_init()
        pyishmem.ishmem_finalize()

    def test_my_pe(self):
        """Test getting PE number."""
        pyishmem.ishmem_init()
        try:
            pe = pyishmem.ishmem_my_pe()
            assert isinstance(pe, int)
            assert pe >= 0
        finally:
            pyishmem.ishmem_finalize()

    def test_n_pes(self):
        """Test getting number of PEs."""
        pyishmem.ishmem_init()
        try:
            n_pes = pyishmem.ishmem_n_pes()
            assert isinstance(n_pes, int)
            assert n_pes >= 1
        finally:
            pyishmem.ishmem_finalize()


class TestIshmemMalloc:
    """Tests for ishmem_malloc and ishmem_free."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Initialize ishmem before each test and finalize after."""
        pyishmem.ishmem_init()
        yield
        pyishmem.ishmem_finalize()

    def test_malloc_basic(self):
        """Test basic memory allocation."""
        size = 1024
        ptr = pyishmem.ishmem_malloc(size)
        assert ptr != 0
        assert isinstance(ptr, int)
        pyishmem.ishmem_free(ptr)

    def test_malloc_various_sizes(self):
        """Test allocation of various sizes."""
        sizes = [1, 64, 256, 1024, 4096, 1024 * 1024]  # 1B to 1MB
        for size in sizes:
            ptr = pyishmem.ishmem_malloc(size)
            assert ptr != 0, f"Failed to allocate {size} bytes"
            pyishmem.ishmem_free(ptr)

    def test_malloc_multiple(self):
        """Test multiple allocations."""
        ptrs = []
        for i in range(10):
            ptr = pyishmem.ishmem_malloc(1024)
            assert ptr != 0
            ptrs.append(ptr)

        # Verify all pointers are unique
        assert len(set(ptrs)) == len(ptrs)

        # Free all
        for ptr in ptrs:
            pyishmem.ishmem_free(ptr)

    def test_free_null(self):
        """Test that freeing null pointer is safe."""
        pyishmem.ishmem_free(0)  # Should not raise

    def test_calloc(self):
        """Test calloc allocation."""
        count = 100
        size = 8
        ptr = pyishmem.ishmem_calloc(count, size)
        assert ptr != 0
        pyishmem.ishmem_free(ptr)

    def test_align(self):
        """Test aligned allocation."""
        alignment = 64
        size = 1024
        ptr = pyishmem.ishmem_align(alignment, size)
        assert ptr != 0
        # Verify alignment
        assert ptr % alignment == 0
        pyishmem.ishmem_free(ptr)


class TestIshmemSync:
    """Tests for synchronization operations."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Initialize ishmem before each test and finalize after."""
        pyishmem.ishmem_init()
        yield
        pyishmem.ishmem_finalize()

    def test_barrier_all(self):
        """Test barrier synchronization."""
        pyishmem.ishmem_barrier_all()  # Should not raise

    def test_sync_all(self):
        """Test sync all."""
        pyishmem.ishmem_sync_all()  # Should not raise

    def test_fence(self):
        """Test fence operation."""
        pyishmem.ishmem_fence()  # Should not raise

    def test_quiet(self):
        """Test quiet operation."""
        pyishmem.ishmem_quiet()  # Should not raise


class TestIshmemConstants:
    """Tests for ishmem constants."""

    def test_team_constants(self):
        """Test team constants are defined."""
        assert hasattr(pyishmem, 'ISHMEM_TEAM_INVALID')
        assert hasattr(pyishmem, 'ISHMEM_TEAM_WORLD')
        assert hasattr(pyishmem, 'ISHMEM_TEAM_SHARED')
        assert pyishmem.ISHMEM_TEAM_INVALID == -1
        assert pyishmem.ISHMEM_TEAM_WORLD == 0
        assert pyishmem.ISHMEM_TEAM_SHARED == 1

    def test_cmp_constants(self):
        """Test comparison constants are defined."""
        assert hasattr(pyishmem, 'ISHMEM_CMP_EQ')
        assert hasattr(pyishmem, 'ISHMEM_CMP_NE')
        assert hasattr(pyishmem, 'ISHMEM_CMP_GT')
        assert hasattr(pyishmem, 'ISHMEM_CMP_GE')
        assert hasattr(pyishmem, 'ISHMEM_CMP_LT')
        assert hasattr(pyishmem, 'ISHMEM_CMP_LE')

    def test_signal_constants(self):
        """Test signal constants are defined."""
        assert hasattr(pyishmem, 'ISHMEM_SIGNAL_SET')
        assert hasattr(pyishmem, 'ISHMEM_SIGNAL_ADD')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

