#pragma once

/**
 * @file vesper.h
 * @brief Vesper Deep Learning Library v1.0 - Main Include Header
 * 
 * A pure C++ deep learning library for HIP/ROCm and CUDA.
 * 
 * Quick Start:
 *   #include <vesper/vesper.h>
 *   using namespace vesper;
 *   
 *   // Create tensors
 *   Tensor x = randn({32, 128}, DType::Float32, Device::HIP, true);
 *   
 *   // Build models
 *   nn::Linear layer(128, 64);
 *   Tensor y = layer.forward(x);
 *   
 *   // Train
 *   auto optimizer = optim::Adam(model.parameters(), 1e-3);
 *   loss.backward();
 *   optimizer.step();
 * 
 * For comprehensive API documentation, see:
 *   #include <vesper/api_reference.h>
 * 
 * Namespaces:
 *   vesper::             Core (Tensor, factories, Device, DType)
 *   vesper::nn::         Neural network modules (Linear, LayerNorm, etc.)
 *   vesper::nn::functional::  Functional API (relu, softmax, cross_entropy_loss)
 *   vesper::ops::        Low-level operations (matmul, sum, exp, etc.)
 *   vesper::optim::      Optimizers (SGD, Adam, AdamW)
 *   vesper::autograd::   Automatic differentiation
 *   vesper::io::         Serialization (save/load)
 */

// Core
#include <vesper/core/tensor.h>
#include <vesper/core/device.h>
#include <vesper/core/dtype.h>
#include <vesper/core/factories.h>
#include <vesper/core/storage.h>
#include <vesper/core/stream.h>

// Autograd
#include <vesper/autograd/engine.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>

// Neural Network Modules
#include <vesper/nn/module.h>
#include <vesper/nn/module_list.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/conv2d.h>
#include <vesper/nn/embedding.h>
#include <vesper/nn/normalization.h>
#include <vesper/nn/activations.h>
#include <vesper/nn/pooling.h>
#include <vesper/nn/init.h>
#include <vesper/nn/functional.h>
#include <vesper/nn/loss.h>

// Transformer Components
#include <vesper/nn/rope.h>
#include <vesper/nn/swiglu.h>
#include <vesper/nn/gqa_attention.h>

// Models
#include <vesper/models/config.h>
#include <vesper/models/transformer_block.h>
#include <vesper/models/transformer.h>

// Operations
#include <vesper/ops/elementwise.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/comparison.h>
#include <vesper/ops/random.h>
#include <vesper/ops/cat.h>
#include <vesper/ops/stack.h>

// Optimizers
#include <vesper/optim/optimizer.h>
#include <vesper/optim/sgd.h>
#include <vesper/optim/adam.h>
#include <vesper/optim/schedulers.h>

// Data Loading
#include <vesper/data/dataloader.h>

// Serialization & I/O
#include <vesper/core/state_dict.h>
#include <vesper/io/io.h>

// Generation
#include <vesper/generation/sampling.h>
#include <vesper/generation/generator.h>
