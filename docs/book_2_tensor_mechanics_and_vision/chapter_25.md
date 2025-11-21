
# Vesper Future Plans - Chapter 24: Pooling Layers (`nn.MaxPool2d`)

## 1. Goal

Implement 2D max pooling, a standard component in Convolutional Neural Networks used to downsample feature maps and create a degree of translational invariance.

## 2. Features

-   **Forward Pass Kernel:** Create a new HIP/CPU kernel that slides a window over the input tensor and, for each window, writes the maximum value to the output tensor.
-   **Backward Pass Logic:** The backward pass for max pooling is unique. The upstream gradient is passed back *only* to the neuron that had the maximum value in the original forward pass window. All other neurons in that window receive a gradient of zero.
-   **Index Storing:** To implement the backward pass efficiently, the forward pass must store the indices of the maximum values (the "argmax") for each window. This state must be saved for the backward pass.

## 3. Why It's Next

Pooling is a near-universal component in CNN architectures, almost always appearing after a convolution and activation layer. It's an essential next step after `Conv2d` for building any standard vision model.
