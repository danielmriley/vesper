#pragma once

#include <vector>
#include <functional>
#include <memory>

namespace vesper::autograd {

class Node; // Forward declaration

// An edge represents a dependency in the graph.
// Uses weak_ptr to prevent reference cycles in the autograd graph.
// Cycles can occur when operations create circular dependencies
// (e.g., x = f(x) patterns where output depends on input).
struct Edge {
    std::weak_ptr<Node> node;  // Changed from shared_ptr to weak_ptr

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
    // Using weak_ptr prevents cycles from keeping nodes alive indefinitely.
    std::vector<Edge> next_edges;
};

} // namespace vesper::autograd
