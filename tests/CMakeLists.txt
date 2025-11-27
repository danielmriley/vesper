# Define the test executable
add_executable(core_tests test_core.cpp)
target_link_libraries(core_tests PRIVATE vesper)
add_test(NAME CoreEnumTests COMMAND core_tests)

# Granular DType tests
add_executable(dtype_tests test_dtype.cpp)
target_link_libraries(dtype_tests PRIVATE vesper)
add_test(NAME DTypeTests COMMAND dtype_tests)

# Granular Device tests
add_executable(device_tests test_device.cpp)
target_link_libraries(device_tests PRIVATE vesper)
add_test(NAME DeviceTests COMMAND device_tests)

# Storage tests
add_executable(storage_tests test_storage.cpp)
target_link_libraries(storage_tests PRIVATE vesper)
add_test(NAME StorageTests COMMAND storage_tests)

# Granular Storage tests
add_executable(storage_granular_tests test_storage_granular.cpp)
target_link_libraries(storage_granular_tests PRIVATE vesper)
add_test(NAME StorageGranularTests COMMAND storage_granular_tests)

# Tensor tests
add_executable(tensor_tests test_tensor.cpp)
target_link_libraries(tensor_tests PRIVATE vesper)
add_test(NAME TensorTests COMMAND tensor_tests)

# Granular Tensor tests
add_executable(tensor_granular_tests test_tensor_granular.cpp)
target_link_libraries(tensor_granular_tests PRIVATE vesper)
add_test(NAME TensorGranularTests COMMAND tensor_granular_tests)

# Zeros Factory tests
add_executable(zeros_tests test_zeros.cpp)
target_link_libraries(zeros_tests PRIVATE vesper)
add_test(NAME ZerosTests COMMAND zeros_tests)

# Tensor Properties tests
add_executable(tensor_props_tests test_tensor_props.cpp)
target_link_libraries(tensor_props_tests PRIVATE vesper)
add_test(NAME TensorPropsTests COMMAND tensor_props_tests)

# Advanced Storage tests
add_executable(storage_advanced_tests test_storage_advanced.cpp)
target_link_libraries(storage_advanced_tests PRIVATE vesper)
add_test(NAME StorageAdvancedTests COMMAND storage_advanced_tests)

# Element-wise Operations tests
add_executable(elementwise_op_tests test_elementwise_ops.cpp)
target_link_libraries(elementwise_op_tests PRIVATE vesper)
add_test(NAME ElementwiseOpTests COMMAND elementwise_op_tests)

# GEMM Kernel tests
add_executable(gemm_kernel_tests test_gemm_kernel.cpp)
target_link_libraries(gemm_kernel_tests PRIVATE vesper)
add_test(NAME GemmKernelTests COMMAND gemm_kernel_tests)

# Matmul Op tests
add_executable(matmul_op_tests test_matmul_op.cpp)
target_link_libraries(matmul_op_tests PRIVATE vesper)
add_test(NAME MatmulOpTests COMMAND matmul_op_tests)

# Comprehensive GEMM tests
add_executable(gemm_comprehensive_tests test_gemm_comprehensive.cpp)
target_link_libraries(gemm_comprehensive_tests PRIVATE vesper)
add_test(NAME GemmComprehensiveTests COMMAND gemm_comprehensive_tests)

# Reduction Op tests
add_executable(reduction_op_tests test_reduction_op.cpp)
target_link_libraries(reduction_op_tests PRIVATE vesper)
add_test(NAME ReductionOpTests COMMAND reduction_op_tests)

# Backend Consistency tests
add_executable(backend_consistency_tests test_backend_consistency.cpp)
target_link_libraries(backend_consistency_tests PRIVATE vesper)
add_test(NAME BackendConsistencyTests COMMAND backend_consistency_tests)

# Autograd tests
add_executable(autograd_tests test_autograd.cpp)
target_link_libraries(autograd_tests PRIVATE vesper)
add_test(NAME AutogradTests COMMAND autograd_tests)

# Autograd Engine tests
add_executable(autograd_engine_tests test_autograd_engine.cpp)
target_link_libraries(autograd_engine_tests PRIVATE vesper)
add_test(NAME AutogradEngineTests COMMAND autograd_engine_tests)

# Autograd Reduction tests
add_executable(autograd_reduction_tests test_autograd_reduction.cpp)
target_link_libraries(autograd_reduction_tests PRIVATE vesper)
add_test(NAME AutogradReductionTests COMMAND autograd_reduction_tests)

# Autograd Matmul tests
add_executable(autograd_matmul_tests test_autograd_matmul.cpp)
target_link_libraries(autograd_matmul_tests PRIVATE vesper)
add_test(NAME AutogradMatmulTests COMMAND autograd_matmul_tests)

# Tensor View tests
add_executable(tensor_view_tests test_tensor_views.cpp)
target_link_libraries(tensor_view_tests PRIVATE vesper)
add_test(NAME TensorViewTests COMMAND tensor_view_tests)

# NN Module tests
add_executable(nn_module_tests test_nn.cpp)
target_link_libraries(nn_module_tests PRIVATE vesper)
add_test(NAME NnModuleTests COMMAND nn_module_tests)

# NN Layer tests
add_executable(nn_layer_tests test_nn_layers.cpp)
target_link_libraries(nn_layer_tests PRIVATE vesper)
add_test(NAME NnLayerTests COMMAND nn_layer_tests)

# Autograd Ops tests
add_executable(autograd_ops_tests test_autograd_ops.cpp)
target_link_libraries(autograd_ops_tests PRIVATE vesper)
add_test(NAME AutogradOpsTests COMMAND autograd_ops_tests)

# Activation tests
add_executable(activation_tests test_activations.cpp)
target_link_libraries(activation_tests PRIVATE vesper)
add_test(NAME ActivationTests COMMAND activation_tests)

# NN Complex tests
add_executable(nn_complex_tests test_nn_complex.cpp)
target_link_libraries(nn_complex_tests PRIVATE vesper)
add_test(NAME NnComplexTests COMMAND nn_complex_tests)

# Linear Grad Check tests
add_executable(linear_grad_check_tests test_linear_grad_check.cpp)
target_link_libraries(linear_grad_check_tests PRIVATE vesper)
add_test(NAME LinearGradCheckTests COMMAND linear_grad_check_tests)

# Autograd Matmul Complex tests
add_executable(autograd_matmul_complex_tests test_autograd_matmul_complex.cpp)
target_link_libraries(autograd_matmul_complex_tests PRIVATE vesper)
add_test(NAME AutogradMatmulComplexTests COMMAND autograd_matmul_complex_tests)

# Activation Edge tests
add_executable(activation_edge_tests test_activations_edge.cpp)
target_link_libraries(activation_edge_tests PRIVATE vesper)
add_test(NAME ActivationEdgeTests COMMAND activation_edge_tests)

# Stress Large tests
add_executable(stress_large_tests test_stress_large.cpp)
target_link_libraries(stress_large_tests PRIVATE vesper)
add_test(NAME StressLargeTests COMMAND stress_large_tests)

# Broadcasting HIP tests
add_executable(broadcasting_hip_tests test_broadcasting_hip.cpp)
target_link_libraries(broadcasting_hip_tests PRIVATE vesper)
add_test(NAME BroadcastingHipTests COMMAND broadcasting_hip_tests)

# NN GPU tests
add_executable(nn_gpu_tests test_nn_gpu.cpp)
target_link_libraries(nn_gpu_tests PRIVATE vesper)
add_test(NAME NnGpuTests COMMAND nn_gpu_tests)

# Div Mean GPU tests
add_executable(div_mean_gpu_tests test_div_mean_gpu.cpp)
target_link_libraries(div_mean_gpu_tests PRIVATE vesper)
add_test(NAME DivMeanGpuTests COMMAND div_mean_gpu_tests)

# Random GPU tests
add_executable(random_gpu_tests test_random_gpu.cpp)
target_link_libraries(random_gpu_tests PRIVATE vesper)
add_test(NAME RandomGpuTests COMMAND random_gpu_tests)

# Loss GPU tests
add_executable(loss_gpu_tests test_loss_gpu.cpp)
target_link_libraries(loss_gpu_tests PRIVATE vesper)
add_test(NAME LossGpuTests COMMAND loss_gpu_tests)

# Optimizer tests
add_executable(optimizer_tests test_optimizer.cpp)
target_link_libraries(optimizer_tests PRIVATE vesper)
add_test(NAME OptimizerTests COMMAND optimizer_tests)

# SGD tests
add_executable(sgd_tests test_sgd.cpp)
target_link_libraries(sgd_tests PRIVATE vesper)
add_test(NAME SgdTests COMMAND sgd_tests)

# Training Loop tests
add_executable(training_loop_tests test_training_loop.cpp)
target_link_libraries(training_loop_tests PRIVATE vesper)
add_test(NAME TrainingLoopTests COMMAND training_loop_tests)

# Data tests
add_executable(data_tests test_data.cpp)
target_link_libraries(data_tests PRIVATE vesper)
add_test(NAME DataTests COMMAND data_tests)

# Stack Op tests
add_executable(stack_op_tests test_stack_op.cpp)
target_link_libraries(stack_op_tests PRIVATE vesper)
add_test(NAME StackOpTests COMMAND stack_op_tests)

# Autograd More Ops tests
add_executable(autograd_more_ops_tests test_autograd_more_ops.cpp)
target_link_libraries(autograd_more_ops_tests PRIVATE vesper)
add_test(NAME AutogradMoreOpsTests COMMAND autograd_more_ops_tests)

# Tensor Slicing tests
add_executable(tensor_slicing_tests test_tensor_slicing.cpp)
target_link_libraries(tensor_slicing_tests PRIVATE vesper)
add_test(NAME TensorSlicingTests COMMAND tensor_slicing_tests)

# Allocator tests
add_executable(tensor_indexing_tests test_tensor_indexing.cpp)
target_link_libraries(tensor_indexing_tests PRIVATE vesper)
add_test(NAME TensorIndexingTests COMMAND tensor_indexing_tests)

# View tests (Testing new view/permute)
add_executable(view_tests test_views.cpp)
target_link_libraries(view_tests PRIVATE vesper)
add_test(NAME ViewTests COMMAND view_tests)

add_executable(view_advanced_tests test_view_advanced.cpp)
target_link_libraries(view_advanced_tests PRIVATE vesper)
add_test(NAME ViewAdvancedTests COMMAND view_advanced_tests)

# DataLoader Shuffle tests
add_executable(dataloader_shuffle_tests test_dataloader_shuffle.cpp)
target_link_libraries(dataloader_shuffle_tests PRIVATE vesper)
add_test(NAME DataloaderShuffleTests COMMAND dataloader_shuffle_tests)

# Broadcasting tests
add_executable(broadcasting_tests test_broadcasting.cpp)
target_link_libraries(broadcasting_tests PRIVATE vesper)
add_test(NAME BroadcastingTests COMMAND broadcasting_tests)

add_executable(broadcasting_rigorous_tests test_broadcasting_rigorous.cpp)
target_link_libraries(broadcasting_rigorous_tests PRIVATE vesper)
add_test(NAME BroadcastingRigorousTests COMMAND broadcasting_rigorous_tests)

add_executable(api_ergonomics_tests test_api_ergonomics.cpp)
target_link_libraries(api_ergonomics_tests PRIVATE vesper)
add_test(NAME ApiErgonomicsTests COMMAND api_ergonomics_tests)

# Conv2d tests
add_executable(conv2d_tests test_conv2d.cpp)
target_link_libraries(conv2d_tests PRIVATE vesper)
add_test(NAME Conv2dTests COMMAND conv2d_tests)

# Pooling tests
add_executable(pooling_tests test_pooling.cpp)
target_link_libraries(pooling_tests PRIVATE vesper)
add_test(NAME PoolingTests COMMAND pooling_tests)

# In-place Safety tests
add_executable(inplace_safety_tests test_inplace_safety.cpp)
target_link_libraries(inplace_safety_tests PRIVATE vesper)
add_test(NAME InplaceSafetyTests COMMAND inplace_safety_tests)

# Init tests
add_executable(init_tests test_init.cpp)
target_link_libraries(init_tests PRIVATE vesper)
add_test(NAME InitTests COMMAND init_tests)

# Optimizer Convergence tests
add_executable(optimizer_convergence_tests test_optimizers.cpp)
target_link_libraries(optimizer_convergence_tests PRIVATE vesper)
add_test(NAME OptimizerConvergenceTests COMMAND optimizer_convergence_tests)

# Optimizer Rigorous tests
add_executable(optimizer_rigorous_tests test_optimizers_rigorous.cpp)
target_link_libraries(optimizer_rigorous_tests PRIVATE vesper)
add_test(NAME OptimizerRigorousTests COMMAND optimizer_rigorous_tests)

# Scheduler tests
add_executable(scheduler_tests test_schedulers.cpp)
target_link_libraries(scheduler_tests PRIVATE vesper)
add_test(NAME SchedulerTests COMMAND scheduler_tests)

# Serialization tests
add_executable(serialization_tests test_serialization.cpp)
target_link_libraries(serialization_tests PRIVATE vesper)
add_test(NAME SerializationTests COMMAND serialization_tests)

# Serialization Rigorous tests
add_executable(serialization_rigorous_tests test_serialization_rigorous.cpp)
target_link_libraries(serialization_rigorous_tests PRIVATE vesper)
add_test(NAME SerializationRigorousTests COMMAND serialization_rigorous_tests)

# Serialization Edge tests
add_executable(serialization_edge_tests test_serialization_edge.cpp)
target_link_libraries(serialization_edge_tests PRIVATE vesper)
add_test(NAME SerializationEdgeTests COMMAND serialization_edge_tests)

# Embedding tests
add_executable(embedding_tests test_embedding.cpp)
target_link_libraries(embedding_tests PRIVATE vesper)
add_test(NAME EmbeddingTests COMMAND embedding_tests)

# Memory Leak tests
add_executable(memory_leak_tests test_memory_leak.cpp)
target_link_libraries(memory_leak_tests PRIVATE vesper)
add_test(NAME MemoryLeakTests COMMAND memory_leak_tests)

# View Autograd tests
add_executable(view_autograd_tests test_view_autograd.cpp)
target_link_libraries(view_autograd_tests PRIVATE vesper)
add_test(NAME ViewAutogradTests COMMAND view_autograd_tests)

# Cast tests
add_executable(cast_tests test_cast.cpp)
target_link_libraries(cast_tests PRIVATE vesper)
add_test(NAME CastTests COMMAND cast_tests)

# More Ops tests
add_executable(more_ops_tests test_more_ops.cpp)
target_link_libraries(more_ops_tests PRIVATE vesper)
add_test(NAME MoreOpsTests COMMAND more_ops_tests)

# Normalization tests
add_executable(normalization_tests test_normalization.cpp)
target_link_libraries(normalization_tests PRIVATE vesper)
add_test(NAME NormalizationTests COMMAND normalization_tests)

# Softmax tests
add_executable(softmax_tests test_softmax.cpp)
target_link_libraries(softmax_tests PRIVATE vesper)
add_test(NAME SoftmaxTests COMMAND softmax_tests)

# Batch GEMM tests
add_executable(batch_gemm_tests test_batch_gemm.cpp)
target_link_libraries(batch_gemm_tests PRIVATE vesper)
add_test(NAME BatchGemmTests COMMAND batch_gemm_tests)

# Batch GEMM comprehensive tests (Chapter 31)
add_executable(batch_gemm_comprehensive_tests test_batch_gemm_comprehensive.cpp)
target_link_libraries(batch_gemm_comprehensive_tests PRIVATE vesper)
add_test(NAME BatchGemmComprehensiveTests COMMAND batch_gemm_comprehensive_tests)

# Attention tests
add_executable(attention_tests test_attention.cpp)
target_link_libraries(attention_tests PRIVATE vesper)
add_test(NAME AttentionTests COMMAND attention_tests)

add_executable(sum_dim_tests test_sum_dim.cpp)
target_link_libraries(sum_dim_tests vesper)
add_test(NAME sum_dim_tests COMMAND sum_dim_tests)

# Dropout tests
add_executable(dropout_tests test_dropout.cpp)
target_link_libraries(dropout_tests PRIVATE vesper)
add_test(NAME DropoutTests COMMAND dropout_tests)

# Robustness tests
add_executable(robustness_tests test_robustness.cpp)
target_link_libraries(robustness_tests PRIVATE vesper)
add_test(NAME RobustnessTests COMMAND robustness_tests)

# Transformer tests
add_executable(transformer_tests test_transformer.cpp)
target_link_libraries(transformer_tests PRIVATE vesper)
add_test(NAME TransformerTests COMMAND transformer_tests)

# Large Linear Regression tests
add_executable(linear_regression_large_tests test_linear_regression_large.cpp)
target_link_libraries(linear_regression_large_tests PRIVATE vesper)
add_test(NAME LinearRegressionLargeTests COMMAND linear_regression_large_tests)

# Linear Slice tests




# Stress Training tests
add_executable(stress_training_tests test_stress_training.cpp)
target_link_libraries(stress_training_tests PRIVATE vesper)
add_test(NAME StressTrainingTests COMMAND stress_training_tests)

# RoPE tests
add_executable(rope_tests test_rope.cpp)
target_link_libraries(rope_tests PRIVATE vesper)
add_test(NAME RopeTests COMMAND rope_tests)

# KV Cache tests
add_executable(kv_cache_tests test_kv_cache.cpp)
target_link_libraries(kv_cache_tests PRIVATE vesper)
add_test(NAME KVCacheTests COMMAND kv_cache_tests)

# Transformer Block tests
add_executable(transformer_block_tests test_transformer_block.cpp)
target_link_libraries(transformer_block_tests PRIVATE vesper)
add_test(NAME TransformerBlockTests COMMAND transformer_block_tests)

# Normalization Gradient tests
add_executable(normalization_grad_tests test_normalization_grad.cpp)
target_link_libraries(normalization_grad_tests PRIVATE vesper)
add_test(NAME NormalizationGradTests COMMAND normalization_grad_tests)

# Normalization Integration tests (Chapter 30)
add_executable(normalization_integration_tests test_normalization_integration.cpp)
target_link_libraries(normalization_integration_tests PRIVATE vesper)
add_test(NAME NormalizationIntegrationTests COMMAND normalization_integration_tests)

# Softmax Comprehensive tests
add_executable(softmax_comprehensive_tests test_softmax_comprehensive.cpp)
target_link_libraries(softmax_comprehensive_tests PRIVATE vesper)
add_test(NAME SoftmaxComprehensiveTests COMMAND softmax_comprehensive_tests)

# Attention Comprehensive tests
add_executable(attention_comprehensive_tests test_attention_comprehensive.cpp)
target_link_libraries(attention_comprehensive_tests PRIVATE vesper)
add_test(NAME AttentionComprehensiveTests COMMAND attention_comprehensive_tests)

# Transformer Components tests
add_executable(transformer_components_tests test_transformer_components.cpp)
target_link_libraries(transformer_components_tests PRIVATE vesper)
add_test(NAME TransformerComponentsTests COMMAND transformer_components_tests)

# Batch GEMM Edge tests
add_executable(batch_gemm_edge_tests test_batch_gemm_edge.cpp)
target_link_libraries(batch_gemm_edge_tests PRIVATE vesper)
add_test(NAME BatchGemmEdgeTests COMMAND batch_gemm_edge_tests)

# Chapter 32 Attention tests
add_executable(attention_chapter32_tests test_attention_chapter32.cpp)
target_link_libraries(attention_chapter32_tests PRIVATE vesper)
add_test(NAME AttentionChapter32Tests COMMAND attention_chapter32_tests)

# Chapter 33.1 Transformer Block tests
add_executable(transformer_block_chapter33_tests test_transformer_block_chapter33.cpp)
target_link_libraries(transformer_block_chapter33_tests PRIVATE vesper)
add_test(NAME TransformerBlockChapter33Tests COMMAND transformer_block_chapter33_tests)

# Chapter 33.2 KV Cache tests
add_executable(kv_cache_chapter33_tests test_kv_cache_chapter33.cpp)
target_link_libraries(kv_cache_chapter33_tests PRIVATE vesper)
add_test(NAME KVCacheChapter33Tests COMMAND kv_cache_chapter33_tests)

# Chapter 33.3 RoPE tests
add_executable(rope_chapter33_tests test_rope_chapter33.cpp)
target_link_libraries(rope_chapter33_tests PRIVATE vesper)
add_test(NAME RoPEChapter33Tests COMMAND rope_chapter33_tests)

# Chapter 33.4 SwiGLU tests
add_executable(swiglu_chapter33_tests test_swiglu_chapter33.cpp)
target_link_libraries(swiglu_chapter33_tests PRIVATE vesper)
add_test(NAME SwiGLUChapter33Tests COMMAND swiglu_chapter33_tests)

# In-place Device Transfer tests (to_() method)
add_executable(inplace_to_tests test_inplace_to.cpp)
target_link_libraries(inplace_to_tests PRIVATE vesper)
add_test(NAME InplaceToTests COMMAND inplace_to_tests)

# ModuleList tests
add_executable(module_list_tests test_module_list.cpp)
target_link_libraries(module_list_tests PRIVATE vesper)
add_test(NAME ModuleListTests COMMAND module_list_tests)

# Chapter 33.5 GQA tests
add_executable(gqa_chapter33_tests test_gqa_chapter33.cpp)
target_link_libraries(gqa_chapter33_tests PRIVATE vesper)
add_test(NAME GQAChapter33Tests COMMAND gqa_chapter33_tests)

# Chapter 33.6 Transformer Model tests
add_executable(transformer_model_chapter33_tests test_transformer_model_chapter33.cpp)
target_link_libraries(transformer_model_chapter33_tests PRIVATE vesper)
add_test(NAME TransformerModelChapter33Tests COMMAND transformer_model_chapter33_tests)

# Chapter 33.7 Generation and Sampling tests
add_executable(generation_chapter33_tests test_generation_chapter33.cpp)
target_link_libraries(generation_chapter33_tests PRIVATE vesper)
add_test(NAME GenerationChapter33Tests COMMAND generation_chapter33_tests)

# Chapter 33.8 Safetensors and Weight Loading tests
add_executable(safetensors_tests test_safetensors.cpp)
target_link_libraries(safetensors_tests PRIVATE vesper)
add_test(NAME SafetensorsTests COMMAND safetensors_tests)
