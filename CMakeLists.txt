cmake_minimum_required(VERSION 3.15)
project(Vesper LANGUAGES CXX C)

# --- 1. Project-wide Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# --- 2. Backend Options ---
# Define options to control which backend to build. Default to HIP ON.
option(USE_HIP "Enable HIP (AMD GPU) backend" ON)
option(USE_CUDA "Enable CUDA (NVIDIA GPU) backend" OFF)
option(USE_CPU "Enable CPU backend" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA toolkit. Enabling CUDA backend.")
    
    # Add a definition for preprocessor directives in the code
    add_compile_definitions(USE_CUDA_BACKEND)
endif()

# --- 2.1. ROCm/HIP Path Configuration ---
# Set ROCm installation path
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

# --- 3. Public Include Directory ---
# This allows users linking against Vesper to find headers.
include_directories(include)

# --- 4. Add the Library Source Code ---
# The core library logic will be defined in the src/ directory.
add_subdirectory(src)

# --- 5. Add Tests and Examples ---
enable_testing()
add_subdirectory(tests)
# add_subdirectory(examples)

# --- 6. Output ---
# Inform the user about the chosen configuration.
message(STATUS "Vesper Build Configuration:")
message(STATUS "  - C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  - USE_HIP: ${USE_HIP}")
message(STATUS "  - USE_CUDA: ${USE_CUDA}")
message(STATUS "  - USE_CPU: ${USE_CPU}")
