#pragma once

#include <vector>
#include <functional>
#include <memory>

namespace vesper::autograd {

class Node; // Forward declaration

// An edge represents a dependency in the graph.
struct Edge {
    std::shared_ptr<Node> node;

    // The function that computes the gradient for this input.
    // It takes the upstream gradient and returns the downstream gradient.
    // For this chapter, we'll keep it simple.
};

// A Node represents an operation in the computational graph.
class Node {
public:
    // The function to execute when backpropagating.
    std::function<void()> backward_fn;
    
    // Pointers to the nodes that are inputs to this operation.
    std::vector<Edge> next_edges;
};

} // namespace vesper::autograd
