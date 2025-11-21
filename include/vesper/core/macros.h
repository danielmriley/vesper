#pragma once

#include <stdexcept>
#include <string>

#define VESPER_CHECK(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error(std::string("VESPER_CHECK failed: ") + (message)); \
    }
