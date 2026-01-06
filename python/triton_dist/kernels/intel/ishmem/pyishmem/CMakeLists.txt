cmake_minimum_required(VERSION 3.12)

# Build pyishmem - Python bindings for Intel SHMEM
project(pyishmem LANGUAGES CXX)

###############################################################################
# Python and pybind11
###############################################################################
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
# Intel SHMEM home directory
if(DEFINED ENV{ISHMEM_HOME})
  set(ISHMEM_HOME "$ENV{ISHMEM_HOME}" CACHE STRING "Intel SHMEM install directory")
else()
  message(FATAL_ERROR "ISHMEM_HOME environment variable is not set. "
                      "Please set it to the Intel SHMEM installation directory.")
endif()

# Intel oneAPI DPC++/C++ compiler
if(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}" CACHE STRING "oneAPI install directory")
  set(CMAKE_CXX_COMPILER "${ONEAPI_ROOT}/compiler/latest/bin/icpx")
else()
  # Try to find icpx in PATH
  find_program(ICPX_COMPILER icpx)
  if(ICPX_COMPILER)
    set(CMAKE_CXX_COMPILER "${ICPX_COMPILER}")
  else()
    message(FATAL_ERROR "Intel DPC++/C++ compiler (icpx) not found. "
                        "Please source the oneAPI environment or set ONEAPI_ROOT.")
  endif()
endif()

message(STATUS "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

###############################################################################
# Find pybind11
###############################################################################
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c"
          "from __future__ import print_function; import os; import pybind11;
print(os.path.dirname(pybind11.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PYBIND11_DIR)
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(STATUS "PYBIND11_DIR: ${PYBIND11_DIR}")
  message(FATAL_ERROR "Pybind11 config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)

###############################################################################
# Intel SHMEM
###############################################################################
set(ISHMEM_INCLUDE_DIR "${ISHMEM_HOME}/include")
set(ISHMEM_LIB_DIR "${ISHMEM_HOME}/lib")

if(NOT EXISTS "${ISHMEM_INCLUDE_DIR}/ishmem.h")
  message(FATAL_ERROR "Intel SHMEM header not found at ${ISHMEM_INCLUDE_DIR}/ishmem.h. "
                      "Please check your ISHMEM_HOME setting.")
endif()

find_library(ISHMEM_LIBRARY
  NAMES ishmem
  PATHS ${ISHMEM_LIB_DIR}
  NO_DEFAULT_PATH)

if(NOT ISHMEM_LIBRARY)
  message(FATAL_ERROR "Intel SHMEM library not found in ${ISHMEM_LIB_DIR}. "
                      "Please check your ISHMEM_HOME setting.")
endif()

message(STATUS "ISHMEM_INCLUDE_DIR: ${ISHMEM_INCLUDE_DIR}")
message(STATUS "ISHMEM_LIBRARY: ${ISHMEM_LIBRARY}")

###############################################################################
# MPI (required by Intel SHMEM)
###############################################################################
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found!")
    message(STATUS "MPI_INCLUDE_DIRS: ${MPI_INCLUDE_DIR}")
    message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
endif()

###############################################################################
# Build the Python module
###############################################################################
pybind11_add_module(_pyishmem src/pyishmem.cc)

target_include_directories(_pyishmem PRIVATE
  ${ISHMEM_INCLUDE_DIR}
)

if(MPI_FOUND)
  target_include_directories(_pyishmem PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

target_link_libraries(_pyishmem PRIVATE
  ${ISHMEM_LIBRARY}
)

if(MPI_FOUND)
  target_link_libraries(_pyishmem PRIVATE ${MPI_CXX_LIBRARIES})
endif()

# SYCL compilation flags for Intel DPC++
target_compile_options(_pyishmem PRIVATE -fsycl)
target_link_options(_pyishmem PRIVATE -fsycl)

set_target_properties(_pyishmem PROPERTIES
  CXX_STANDARD 17
  INTERPROCEDURAL_OPTIMIZATION FALSE
  OUTPUT_NAME "_pyishmem"
)

###############################################################################
# Installation
###############################################################################
install(TARGETS _pyishmem
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(DIRECTORY python/pyishmem/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/python/pyishmem
  FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi"
)

