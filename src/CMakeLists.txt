# --- 1. Define the library target ---
# We will build Vesper as a static library for simplicity.
add_library(vesper STATIC)

# --- 2. Find and Link Backend Libraries ---
if(USE_HIP)
    # Find HIP - we need it for compilation but NOT for exported targets
    find_package(hip REQUIRED)
    if(hip_FOUND)
        message(STATUS "Found HIP runtime. Enabling HIP support.")
        target_compile_definitions(vesper PUBLIC USE_HIP_BACKEND)
        
        # Get HIP include directories and add them directly (not via target)
        get_target_property(HIP_INCLUDE_DIRS hip::device INTERFACE_INCLUDE_DIRECTORIES)
        if(HIP_INCLUDE_DIRS)
            target_include_directories(vesper PRIVATE ${HIP_INCLUDE_DIRS})
        endif()
        
        # Get HIP compile options and add them directly for HIP source files only
        get_target_property(HIP_COMPILE_OPTIONS hip::device INTERFACE_COMPILE_OPTIONS)
        # Note: hipcc already knows these flags, so we don't need to set them
        
        # For consumers: link directly to the HIP runtime shared library
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
        find_library(AMDHIP64_LIBRARY amdhip64 PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
        if(AMDHIP64_LIBRARY)
            # Use INTERFACE so consumers get this library linked
            target_link_libraries(vesper PUBLIC ${AMDHIP64_LIBRARY})
        else()
            message(WARNING "Could not find amdhip64 library - consumers may need to link manually")
        endif()
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
    autograd/checkpoint.cpp
    nn/module.cpp
    nn/init.cpp
    nn/embedding.cpp
    nn/linear.cpp
    nn/conv2d.cpp
    nn/pooling.cpp
    nn/functional.cpp
    nn/normalization.cpp
    nn/transformer.cpp
    nn/rope.cpp
    nn/swiglu.cpp
    nn/gqa_attention.cpp
    nn/amp.cpp
    nn/utils.cpp
    models/config.cpp
    models/transformer_block.cpp
    models/transformer.cpp
    generation/sampling.cpp
    generation/generator.cpp
    generation/beam_search.cpp
    io/safetensors.cpp
    io/safetensors_writer.cpp
    io/weight_mapper.cpp
    io/model_loader.cpp
    data/dataloader.cpp
    optim/optimizer.cpp
    optim/sgd.cpp
    optim/adam.cpp
    optim/lion.cpp
    optim/schedulers.cpp
    optim/grad_scaler.cpp
    serialization.cpp
    ops/elementwise.cpp
    ops/gemm.cpp
    ops/reduction.cpp
    ops/comparison.cpp
    ops/random.cpp
    ops/cast.cpp
    ops/view_ops.cpp
    ops/stack.cpp
    ops/cat.cpp
    ops/im2col.cpp
    ops/pooling.cpp
    ops/embedding.cpp
    ops/normalization.cpp
    ops/attention_ops.cpp
    ops/flash_attention.cpp
    ops/sampling_ops.cpp
    ops/index_ops.cpp
    ops/fused_ops.cpp
    ops/cpu/elementwise_cpu.cpp
    ops/cpu/gemm_cpu.cpp
    ops/cpu/reduction_cpu.cpp
    ops/cpu/im2col_cpu.cpp
    ops/cpu/pooling_cpu.cpp
    ops/cpu/cast_cpu.cpp
    ops/cpu/copy_cpu.cpp
    ops/cuda/activation.cu
    ops/cuda/embedding.cu
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
        ops/hip/embedding.hip
        ops/hip/cast.hip
        ops/hip/normalization.hip
        ops/hip/rope.hip
        ops/hip/attention_ops.hip
        ops/hip/flash_attention.hip
        ops/hip/sampling.hip
        ops/hip/cat.hip
        ops/hip/index_ops.hip
        ops/hip/fused_ops.hip
    )
    # Tell CMake to treat .hip files as C++ source
    set_source_files_properties(
        ops/hip/elementwise.hip 
        ops/hip/gemm.hip 
        ops/hip/reduction.hip 
        ops/hip/activation.hip 
        ops/hip/comparison.hip 
        ops/hip/random.hip 
        ops/hip/copy.hip 
        ops/hip/im2col.hip 
        ops/hip/pooling.hip 
        ops/hip/normalization.hip 
        ops/hip/cast.hip
        ops/hip/embedding.hip
        ops/hip/rope.hip
        ops/hip/attention_ops.hip
        ops/hip/flash_attention.hip
        ops/hip/sampling.hip
        ops/hip/cat.hip
        ops/hip/index_ops.hip
        ops/hip/fused_ops.hip
        PROPERTIES LANGUAGE CXX
    )
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
