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
    message(STATUS "CUDA backend enabled.")
    target_compile_definitions(vesper PUBLIC USE_CUDA_BACKEND)
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
    core/broadcasting.cpp
    core/allocator.cpp
    core/stream.cpp
    core/reference_ops.cpp
    autograd/engine.cpp
    autograd/guard.cpp
    nn/module.cpp
    nn/linear.cpp
    nn/functional.cpp
    ops/comparison.cpp
    ops/elementwise.cpp
    ops/gemm.cpp
    ops/reduction.cpp
    ops/random.cpp
    ops/stack.cpp
    ops/view_ops.cpp
    ops/cast.cpp
    ops/cpu/gemm_cpu.cpp
    ops/cpu/elementwise_cpu.cpp
    ops/cpu/reduction_cpu.cpp
    ops/cpu/copy_cpu.cpp
    ops/cpu/cast_cpu.cpp
    ops/cpu/im2col_cpu.cpp
    ops/cpu/pooling_cpu.cpp
    ops/im2col.cpp
    ops/pooling.cpp
    nn/conv2d.cpp
    nn/pooling.cpp
    optim/optimizer.cpp
    optim/sgd.cpp
    data/dataloader.cpp
)

if(USE_HIP)
    target_sources(vesper PRIVATE
        ops/hip/elementwise.hip
        ops/hip/gemm.hip
        ops/hip/reduction.hip
        ops/hip/activation.hip
        ops/hip/comparison.hip
        ops/hip/random.hip
        ops/hip/copy.hip
        ops/hip/im2col.hip
        ops/hip/pooling.hip
    )
    # Tell CMake to treat .hip files as C++ source
    set_source_files_properties(ops/hip/elementwise.hip ops/hip/gemm.hip ops/hip/reduction.hip ops/hip/activation.hip ops/hip/comparison.hip ops/hip/random.hip ops/hip/copy.hip ops/hip/im2col.hip ops/hip/pooling.hip PROPERTIES LANGUAGE CXX)
endif()

if(USE_CUDA)
    file(GLOB CUDA_SOURCES "ops/cuda/*.cu")
    target_sources(vesper PRIVATE ${CUDA_SOURCES})
    # Link against the CUDA runtime library
    target_link_libraries(vesper PUBLIC CUDA::cudart)
endif()

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
