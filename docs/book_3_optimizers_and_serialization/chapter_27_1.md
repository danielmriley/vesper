
# Vesper Future Plans - Chapter 25: The Adam Optimizer (`optim.Adam`)

## 1. Goal

Implement the Adam (Adaptive Moment Estimation) optimizer, one of the most popular and effective optimization algorithms for training deep neural networks.

## 2. Features

-   **Stateful Optimizer:** Unlike SGD, Adam is a "stateful" optimizer. This chapter will require extending the `Optimizer` base class or creating a new pattern to handle optimizer state.
-   **Moment Tracking:** For each model parameter it manages, Adam must store and update two "moments":
    1.  `m`: The moving average of the gradients.
    2.  `v`: The moving average of the squared gradients.
    These will be stored as separate `Tensor`s, with the same shape as the parameter they correspond to.
-   **`step()` Implementation:** The `step()` method will be significantly more complex than SGD's. It will involve several element-wise operations (multiplication, addition, square root, division) to update the `m` and `v` moments and then compute the final parameter update using these adaptive moments.
-   **Bias Correction:** The implementation must include bias-correction for the first few steps of training to counteract the fact that the moments are initialized to zero.

## 3. Why It's Next

While SGD is a great starting point, Adam often converges much faster and is less sensitive to the choice of learning rate. It is the de-facto standard optimizer for many deep learning tasks, making it a high-priority addition to move Vesper from a proof-of-concept to a practical library.
