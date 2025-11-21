#pragma once

namespace vesper::autograd {

extern thread_local bool grad_mode_enabled;

struct NoGradGuard {
    NoGradGuard() {
        prev_mode = grad_mode_enabled;
        grad_mode_enabled = false;
    }
    ~NoGradGuard() {
        grad_mode_enabled = prev_mode;
    }
    bool prev_mode;
};

struct EnableGradGuard {
    EnableGradGuard() {
        prev_mode = grad_mode_enabled;
        grad_mode_enabled = true;
    }
    ~EnableGradGuard() {
        grad_mode_enabled = prev_mode;
    }
    bool prev_mode;
};

} // namespace vesper::autograd
