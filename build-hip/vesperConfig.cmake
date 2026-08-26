
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was vesperConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# Vesper CMake Configuration File
# This file is used by find_package(vesper) to locate the library

include(CMakeFindDependencyMacro)

# Find HIP if it was used to build Vesper
if(ON)
    set(ROCM_PATH "/opt/rocm-7.2.4" CACHE PATH "Path to ROCm installation")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
    find_dependency(hip REQUIRED)
endif()

# Find CUDA if it was used to build Vesper
if(OFF)
    find_dependency(CUDAToolkit REQUIRED)
endif()

# Include the targets file
include("${CMAKE_CURRENT_LIST_DIR}/vesperTargets.cmake")

# Check that all required components were found
check_required_components(vesper)

# Provide some helpful variables
set(VESPER_FOUND TRUE)
set(VESPER_VERSION 1.0.0)
set(VESPER_USE_HIP ON)
set(VESPER_USE_CUDA OFF)
set(VESPER_USE_CPU OFF)

message(STATUS "Found Vesper ${VESPER_VERSION}")
if(VESPER_USE_HIP)
    message(STATUS "  - HIP backend enabled")
endif()
if(VESPER_USE_CUDA)
    message(STATUS "  - CUDA backend enabled")
endif()
if(VESPER_USE_CPU)
    message(STATUS "  - CPU backend enabled")
endif()
