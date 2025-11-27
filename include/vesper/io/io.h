#pragma once
/// @file io.h
/// @brief Convenience header including all I/O functionality.
///
/// Include this single header to get access to:
/// - Safetensors reading and writing
/// - Weight mapping for model format conversion  
/// - High-level model loading
///
/// Chapter 33.8: Loading Pre-trained Weights

#include <vesper/io/safetensors.h>
#include <vesper/io/safetensors_writer.h>
#include <vesper/io/weight_mapper.h>
#include <vesper/io/model_loader.h>
