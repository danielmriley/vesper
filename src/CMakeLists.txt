# --- 1. Define the library target ---
# We will build Vesper as a static library for simplicity.
add_library(vesper STATIC)

# --- 2. Find and Link Backend Libraries ---
if(USE_HIP)
    # Find the HIP package provided by ROCm.
    # Changed from amdhip-runtime to hip based on file system exploration
    find_package(hip REQUIRED)
    if(hip_FOUND)
        message(STATUS "Found HIP runtime. Enabling HIP support.")
        # Add a definition to use in preprocessor directives.
        target_compile_definitions(vesper PUBLIC USE_HIP_BACKEND)
        # Link the library to the HIP runtime.
        # Usually hip::device is the target to link against for runtime
        target_link_libraries(vesper PUBLIC hip::device)
    endif()
endif()

if(USE_CUDA)
    # Stub for CUDA
    message(WARNING "CUDA backend is not yet implemented.")
    target_compile_definitions(vesper PUBLIC USE_CUDA_BACKEND)
    # In the future, you would find and link CUDA here.
    # find_package(CUDA REQUIRED)
    # target_link_libraries(vesper PUBLIC CUDA::cudart)
endif()

if(USE_CPU)
    # Stub for CPU
    message(STATUS "CPU backend stub enabled.")
    target_compile_definitions(vesper PUBLIC USE_CPU_BACKEND)
endif()

# --- 3. Add Source Files ---
# As we create files, we will add them here.
target_sources(vesper PRIVATE
    core/storage.cpp
    core/tensor.cpp
    autograd/engine.cpp
    ops/elementwise.cpp
    ops/gemm.cpp
    ops/reduction.cpp
    ops/hip/elementwise.hip
    ops/hip/gemm.hip
    ops/hip/reduction.hip
)

# Tell CMake to treat .hip files as C++ source
set_source_files_properties(ops/hip/elementwise.hip ops/hip/gemm.hip ops/hip/reduction.hip PROPERTIES LANGUAGE CXX)

# --- 4. Expose Public Headers ---
# This makes headers in `include/vesper` available to the `vesper` target
# and to any other target that links against it.
target_include_directories(vesper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
