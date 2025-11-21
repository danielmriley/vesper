
# Vesper Future Plans - Chapter 22: Broadcasting and Advanced Indexing

## 1. Goal

Implement broadcasting, the mechanism that allows element-wise operations on tensors of different but compatible shapes. This is a critical feature that unlocks more expressive and efficient computations, such as adding a bias vector to a matrix of outputs.

## 2. Features

-   **Broadcasting Logic:** Implement the core logic that determines the output shape and the required strides for broadcasting two tensors. This typically involves prepending dimensions of size 1 to the smaller tensor and then "expanding" dimensions of size 1 to match the larger tensor's corresponding dimension without copying data.
-   **Kernel Modification:** Update the element-wise kernels (`add`, `mul`, etc.) to accept stride information for each input tensor. The index calculation `out[idx] = op(a[idx], b[idx])` will be replaced with `out[out_idx] = op(a[a_idx], b[b_idx])`, where `a_idx` and `b_idx` are calculated from the output index and the respective tensor's strides.
-   **Bias Addition:** With broadcasting, the `add` operation in the `nn.Linear` layer's forward pass can finally be implemented correctly (`output = ops.add(output, bias)`).
-   **Advanced Indexing:** Implement more advanced indexing to select or assign values to arbitrary slices of a tensor, beyond the simple `slice(i)` method.

## 3. Why It's Next

Broadcasting is a fundamental feature of modern tensor libraries. Its absence is a major limitation in the current MVP, preventing natural expression of common operations like bias addition. Implementing it is the next logical step to make the library more powerful and user-friendly.
