cmake_minimum_required(VERSION 3.16)
project(vesper VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# v1 GPU: Radeon AI Pro R9700 (RDNA 4). See docs/TARGET.md.
set(VESPER_HIP_ARCH "gfx1201" CACHE STRING "HIP GPU architecture (v1 = RDNA4 R9700)")
option(VESPER_USE_HIP "Build the gfx1201 HIP backend" OFF)

if(VESPER_USE_HIP AND NOT VESPER_HIP_ARCH STREQUAL "gfx1201")
    message(WARNING "v1 is tuned for gfx1201; you set VESPER_HIP_ARCH=${VESPER_HIP_ARCH}")
endif()

set(VESPER_SOURCES
    src/buffer.cpp
    src/config.cpp
    src/weights.cpp
    src/kv_cache.cpp
    src/kernels_cpu.cpp
    src/kernels_dispatch.cpp
    src/hip_runtime.cpp
    src/engine.cpp
    src/gguf.cpp
    src/gguf_write.cpp
    src/q8.cpp
    src/q4k.cpp
    src/weight.cpp
    src/model_io.cpp
    src/gdn.cpp
    src/report.cpp
    src/tokenizer.cpp
)

if(VESPER_USE_HIP)
    if(CMAKE_VERSION VERSION_LESS 3.21)
        message(FATAL_ERROR "VESPER_USE_HIP requires CMake 3.21 or newer")
    endif()
    if(DEFINED ENV{ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
    elseif(EXISTS "/opt/rocm")
        list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    endif()
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_ARCHITECTURES "${VESPER_HIP_ARCH}" CACHE STRING "HIP GPU architectures" FORCE)
    set(GPU_TARGETS "${VESPER_HIP_ARCH}" CACHE STRING "ROCm GPU targets" FORCE)
    find_package(hip REQUIRED CONFIG)
    list(APPEND VESPER_SOURCES src/kernels_hip.hip)
else()
    list(APPEND VESPER_SOURCES src/rdna4_stub.cpp)
endif()

add_library(vesper ${VESPER_SOURCES})
target_include_directories(vesper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(vesper PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
)

if(VESPER_USE_HIP)
    target_compile_definitions(vesper PUBLIC VESPER_USE_HIP=1)
    target_link_libraries(vesper PUBLIC hip::host)
    set_source_files_properties(src/kernels_hip.hip PROPERTIES LANGUAGE HIP)
    message(STATUS "Vesper HIP: ON  arch=${VESPER_HIP_ARCH}")
else()
    message(STATUS "Vesper HIP: OFF  (CPU oracle only)")
endif()

add_executable(vesper-infer src/cli.cpp)
target_link_libraries(vesper-infer PRIVATE vesper)
target_compile_options(vesper-infer PRIVATE -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_executable(vesper-tests tests/test_main.cpp)
target_link_libraries(vesper-tests PRIVATE vesper)
target_compile_options(vesper-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME vesper-tests COMMAND vesper-tests)
