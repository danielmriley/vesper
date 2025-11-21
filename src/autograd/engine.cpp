#include <vesper/autograd/engine.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

namespace vesper::autograd {

void Engine::backward(Tensor& root) {
    if (!root.requires_grad()) {
        throw std::runtime_error("Cannot call backward on a tensor that does not require grad.");
    }

    // 1. Initialize root gradient
    // We set it to ones. In a real engine, we might check if it's already set.
    root.grad() = full(root.shape(), root.dtype(), root.device(), 1.0f);

    if (!root.grad_node) {
        // Root is a leaf. Nothing to backpropagate.
        return;
    }

    // 2. Compute dependencies (in-degree in backward graph)
    std::unordered_map<Node*, int> dependencies;
    std::unordered_set<Node*> seen;
    std::queue<std::shared_ptr<Node>> queue;

    queue.push(root.grad_node);
    seen.insert(root.grad_node.get());

    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();

        for (const auto& edge : node->next_edges) {
            if (edge.node) {
                dependencies[edge.node.get()]++;
                if (seen.find(edge.node.get()) == seen.end()) {
                    seen.insert(edge.node.get());
                    queue.push(edge.node);
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
            if (edge.node) {
                dependencies[edge.node.get()]--;
                if (dependencies[edge.node.get()] == 0) {
                    ready_queue.push(edge.node);
                }
            }
        }
    }
}

} // namespace vesper::autograd
