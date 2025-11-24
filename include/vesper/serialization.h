#pragma once
#include <vesper/core/state_dict.h>
#include <vesper/nn/module.h>
#include <string>

namespace vesper {

// Saves the state dictionary to a file
void save(const StateDict& state_dict, const std::string& filename);

// Saves the module's state dictionary
void save(const nn::Module& module, const std::string& filename);

// Loads a state dictionary from a file
StateDict load(const std::string& filename);

// Loads a module's state from a file
void load(nn::Module& module, const std::string& filename);

} // namespace vesper
