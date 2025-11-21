#include <vesper/autograd/guard.h>

namespace vesper::autograd {

thread_local bool grad_mode_enabled = true;

} // namespace vesper::autograd
