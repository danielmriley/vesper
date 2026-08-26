---
applyTo: "**/*.{cpp,h,cu,hip}"
excludeAgent: ["code-reviewer"]
---

# Code Writer Instructions

## Style Guide
-   **Standard**: C++17.
-   **Naming**: `Snake_case` for functions/variables, `PascalCase` for classes.
-   **Namespace**: All code in `namespace vesper`.
-   **Headers**: Use `#pragma once`.
-   **Error Handling**: Use `VESPER_CHECK(condition, msg)` macro.

## Implementation Rules
-   **HIP First**: Always implement the HIP path first. Use `#if USE_HIP_BACKEND`.
-   **No Raw Pointers**: Use `std::shared_ptr` or `std::unique_ptr` where possible, except for low-level `Storage` data access.
-   **Const Correctness**: Mark methods `const` if they don't modify state.
-   **Modularity**: Keep files small (<500 LoC). One class per file usually.

## GPU Kernels
-   Write kernels in `.hip` files.
-   Check `hipError_t` for every HIP call.
-   Use `hipLaunchKernelGGL`.
