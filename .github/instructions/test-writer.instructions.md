---
applyTo: "tests/**"
excludeAgent: ["code-reviewer"]
---

# Test Writer Instructions

## Philosophy
-   **Granularity**: Test one thing at a time.
-   **Coverage**: Test edge cases (empty tensors, large tensors, zero sizes).
-   **Backend**: Always guard GPU tests with `#if USE_HIP_BACKEND`.

## Structure
-   Use `assert()` for simple checks.
-   Print "Testing [Feature]..." at start.
-   Print "[Feature] passed!" at end.
-   Add new tests to `tests/CMakeLists.txt`.

## Verification
-   For GPU tests, use `hipMemcpy` to verify results on host.
-   Do not assume data is zero-initialized unless `zeros()` was used.
