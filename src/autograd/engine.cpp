#include <vesper/autograd/engine.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

namespace vesper::autograd {

void Engine::backward(Tensor& root, const Tensor& gradient) {
    if (!root.requires_grad()) {
        throw std::runtime_error("Cannot call backward on a tensor that does not require grad.");
    }

    // Disable gradient tracking during backward pass to prevent reference cycles
    // (Forward Tensor -> Grad Tensor -> Grad Node -> Forward Tensor)
    // This corresponds to create_graph=False in PyTorch.
    NoGradGuard guard;

    // 1. Initialize root gradient
    if (gradient.storage_use_count() > 0) {
        if (gradient.shape() != root.shape()) {
             throw std::runtime_error("Gradient shape mismatch in backward.");
        }
        root.grad() = gradient;
    } else {
        // Default to ones
        root.grad() = full(root.shape(), root.dtype(), root.device(), 1.0f);
    }

    if (!root.grad_node) {
        // Root is a leaf. Nothing to backpropagate.
        return;
    }

    // 2. Compute dependencies (in-degree in backward graph)
    // Note: edges now use weak_ptr to prevent cycles, so we lock() to access
    std::unordered_map<Node*, int> dependencies;
    std::unordered_set<Node*> seen;
    std::queue<std::shared_ptr<Node>> queue;

    queue.push(root.grad_node);
    seen.insert(root.grad_node.get());

    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();

        for (const auto& edge : node->next_edges) {
            // Lock the weak_ptr to get shared_ptr
            auto edge_node = edge.node.lock();
            if (edge_node) {
                dependencies[edge_node.get()]++;
                if (seen.find(edge_node.get()) == seen.end()) {
                    seen.insert(edge_node.get());
                    queue.push(edge_node);
                }
            }
        }
    }

    // 3. Topological Sort Execution
    std::queue<std::shared_ptr<Node>> ready_queue;
    ready_queue.push(root.grad_node);

    while (!ready_queue.empty()) {
        auto node = ready_queue.front();
        ready_queue.pop();

        // Execute backward function
        if (node->backward_fn) {
            node->backward_fn();
        }

        // Propagate to children
        for (const auto& edge : node->next_edges) {
            auto edge_node = edge.node.lock();
            if (edge_node) {
                dependencies[edge_node.get()]--;
                if (dependencies[edge_node.get()] == 0) {
                    ready_queue.push(edge_node);
                }
            }
        }
    }
}

} // namespace vesper::autograd
