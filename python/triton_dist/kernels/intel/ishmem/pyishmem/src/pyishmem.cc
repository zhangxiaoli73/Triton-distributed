/* Copyright 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Python bindings for Intel SHMEM (ishmem) library.
 * Provides ishmem_malloc and ishmem_free APIs for Python.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Intel SHMEM header
#include <ishmem.h>

namespace py = pybind11;

/**
 * Python module for Intel SHMEM bindings
 */
PYBIND11_MODULE(_pyishmem, m) {
    m.doc() = "Python bindings for Intel SHMEM (ishmem) library";

    // Library initialization and finalization
    m.def("ishmem_init", []() {
        ishmem_init();
    }, "Initialize the Intel SHMEM library");

    m.def("ishmem_finalize", []() {
        ishmem_finalize();
    }, "Finalize the Intel SHMEM library");

    // Query routines
    m.def("ishmem_my_pe", []() -> int {
        return ishmem_my_pe();
    }, "Return the PE number of the calling process");

    m.def("ishmem_n_pes", []() -> int {
        return ishmem_n_pes();
    }, "Return the total number of PEs");

    // Memory management routines
    m.def("ishmem_malloc", [](size_t size) -> intptr_t {
        void *ptr = ishmem_malloc(size);
        if (ptr == nullptr) {
            throw std::runtime_error("ishmem_malloc failed: unable to allocate " +
                                     std::to_string(size) + " bytes");
        }
        return reinterpret_cast<intptr_t>(ptr);
    }, py::arg("size"),
    "Allocate symmetric memory from the Intel SHMEM heap.\n\n"
    "Args:\n"
    "    size: Number of bytes to allocate\n\n"
    "Returns:\n"
    "    Pointer to the allocated memory as an integer\n\n"
    "Raises:\n"
    "    RuntimeError: If allocation fails");

    m.def("ishmem_free", [](intptr_t ptr) {
        if (ptr == 0) {
            return;  // Allow freeing null pointer (no-op)
        }
        ishmem_free(reinterpret_cast<void*>(ptr));
    }, py::arg("ptr"),
    "Free symmetric memory previously allocated by ishmem_malloc.\n\n"
    "Args:\n"
    "    ptr: Pointer to memory to free (as integer)");

    m.def("ishmem_calloc", [](size_t count, size_t size) -> intptr_t {
        void *ptr = ishmem_calloc(count, size);
        if (ptr == nullptr) {
            throw std::runtime_error("ishmem_calloc failed: unable to allocate " +
                                     std::to_string(count * size) + " bytes");
        }
        return reinterpret_cast<intptr_t>(ptr);
    }, py::arg("count"), py::arg("size"),
    "Allocate zero-initialized symmetric memory.\n\n"
    "Args:\n"
    "    count: Number of elements to allocate\n"
    "    size: Size of each element in bytes\n\n"
    "Returns:\n"
    "    Pointer to the allocated memory as an integer");

    m.def("ishmem_align", [](size_t alignment, size_t size) -> intptr_t {
        void *ptr = ishmem_align(alignment, size);
        if (ptr == nullptr) {
            throw std::runtime_error("ishmem_align failed: unable to allocate " +
                                     std::to_string(size) + " bytes with alignment " +
                                     std::to_string(alignment));
        }
        return reinterpret_cast<intptr_t>(ptr);
    }, py::arg("alignment"), py::arg("size"),
    "Allocate aligned symmetric memory.\n\n"
    "Args:\n"
    "    alignment: Alignment requirement in bytes\n"
    "    size: Number of bytes to allocate\n\n"
    "Returns:\n"
    "    Pointer to the allocated memory as an integer");

    // Pointer query
    m.def("ishmem_ptr", [](intptr_t dest, int pe) -> intptr_t {
        void *ptr = ishmem_ptr(reinterpret_cast<const void*>(dest), pe);
        return reinterpret_cast<intptr_t>(ptr);
    }, py::arg("dest"), py::arg("pe"),
    "Get a pointer to a symmetric data object on a specified PE.\n\n"
    "Args:\n"
    "    dest: Pointer to symmetric data object\n"
    "    pe: Target PE number\n\n"
    "Returns:\n"
    "    Pointer to the data on the target PE");

    // Synchronization routines
    m.def("ishmem_barrier_all", []() {
        ishmem_barrier_all();
    }, "Barrier synchronization across all PEs");

    m.def("ishmem_sync_all", []() {
        ishmem_sync_all();
    }, "Synchronize all PEs");

    m.def("ishmem_fence", []() {
        ishmem_fence();
    }, "Ensure ordering of operations");

    m.def("ishmem_quiet", []() {
        ishmem_quiet();
    }, "Wait for completion of all outstanding operations");

    // Team constants
    m.attr("ISHMEM_TEAM_INVALID") = py::int_(ISHMEM_TEAM_INVALID);
    m.attr("ISHMEM_TEAM_WORLD") = py::int_(ISHMEM_TEAM_WORLD);
    m.attr("ISHMEM_TEAM_SHARED") = py::int_(ISHMEM_TEAM_SHARED);

    // Comparison constants
    m.attr("ISHMEM_CMP_EQ") = py::int_(ISHMEM_CMP_EQ);
    m.attr("ISHMEM_CMP_NE") = py::int_(ISHMEM_CMP_NE);
    m.attr("ISHMEM_CMP_GT") = py::int_(ISHMEM_CMP_GT);
    m.attr("ISHMEM_CMP_GE") = py::int_(ISHMEM_CMP_GE);
    m.attr("ISHMEM_CMP_LT") = py::int_(ISHMEM_CMP_LT);
    m.attr("ISHMEM_CMP_LE") = py::int_(ISHMEM_CMP_LE);

    // Signal constants
    m.attr("ISHMEM_SIGNAL_SET") = py::int_(ISHMEM_SIGNAL_SET);
    m.attr("ISHMEM_SIGNAL_ADD") = py::int_(ISHMEM_SIGNAL_ADD);
}

