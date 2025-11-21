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


