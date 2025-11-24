# Chapter 28: Serialization (`state_dict`)

## 1. Introduction

Serialization allows us to save a trained model to disk and load it later. This is essential for:
- **Checkpointing**: Saving progress during long training runs.
- **Inference**: Loading a trained model for deployment.
- **Transfer Learning**: Fine-tuning a pre-trained model.

## 2. Design: The `state_dict`

We will adopt the `state_dict` pattern used by PyTorch. A `state_dict` is simply a map from parameter names to tensors.

```cpp
using StateDict = std::map<std::string, Tensor>;
```

### Module Interface
```cpp
class Module {
public:
    // Returns a map of all parameters (e.g., "layer1.weight" -> Tensor)
    virtual StateDict state_dict() const;
    
    // Loads parameters from a map
    virtual void load_state_dict(const StateDict& state_dict, bool strict=true);
};
```

## 3. File Format

We need a binary format to store the `StateDict` on disk. We will use a simple custom format:

1.  **Header (JSON)**: Contains metadata (version, list of tensors, shapes, dtypes, byte offsets).
2.  **Binary Blob**: Contiguous raw bytes of tensor data.

Example Header:
```json
{
  "version": "1.0",
  "tensors": {
    "fc1.weight": {"dtype": "float32", "shape": [256, 784], "offset": 0, "size": 802816},
    "fc1.bias":   {"dtype": "float32", "shape": [256],      "offset": 802816, "size": 1024}
  }
}
```

## 4. Implementation Plan

### `vesper::save`
1.  Call `model.state_dict()` to get all tensors.
2.  Serialize metadata to a JSON string.
3.  Write JSON length + JSON string to file.
4.  Iterate through tensors, move data to CPU (if on GPU), and write raw bytes to file.

### `vesper::load`
1.  Read file header.
2.  Parse JSON to get tensor info.
3.  Create a `StateDict` with empty tensors of correct shapes.
4.  Read raw bytes into these tensors.
5.  Call `model.load_state_dict(loaded_dict)`.

## 5. Usage Example

```cpp
auto model = MyModel();

// Save
vesper::save(model, "checkpoint.vsp");

// Load
auto model2 = MyModel();
vesper::load(model2, "checkpoint.vsp");
```

## 6. Testing Strategy

1.  **Round-trip Test**: Initialize a model with random weights, save it, load it into a new model, and assert that `model1.weight == model2.weight`.
2.  **Device Handling**: Save a GPU model, load it onto a CPU model (and vice versa).
3.  **Strictness**: Test `load_state_dict` with missing keys or extra keys to verify strict/non-strict behavior.
