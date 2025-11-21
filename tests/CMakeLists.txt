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


