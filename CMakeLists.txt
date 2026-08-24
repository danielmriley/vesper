cmake_minimum_required(VERSION 3.16)
project(vesper VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# v1 GPU: Radeon AI Pro R9700 (RDNA 4). See docs/TARGET.md.
set(VESPER_HIP_ARCH "gfx1201" CACHE STRING "HIP GPU architecture (v1 = RDNA4 R9700)")
option(VESPER_USE_HIP "Build the HIP backend" OFF)

add_library(vesper
    src/buffer.cpp
    src/config.cpp
    src/weights.cpp
    src/kv_cache.cpp
    src/kernels_cpu.cpp
    src/engine.cpp
)
target_include_directories(vesper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(vesper PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(vesper-infer src/cli.cpp)
target_link_libraries(vesper-infer PRIVATE vesper)
target_compile_options(vesper-infer PRIVATE -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_executable(vesper-tests tests/test_main.cpp)
target_link_libraries(vesper-tests PRIVATE vesper)
target_compile_options(vesper-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME vesper-tests COMMAND vesper-tests)
