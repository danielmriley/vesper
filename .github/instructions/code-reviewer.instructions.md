---
applyTo: "**/*.{cpp,h,cu,hip}"
excludeAgent: ["code-writer"]
---

# Code Reviewer Instructions

## What to Look For
1.  **Memory Leaks**: Ensure `hipFree` or `delete[]` is called.
2.  **Error Handling**: Are HIP calls checked? Is `VESPER_CHECK` used?
3.  **Thread Safety**: Are shared resources protected?
4.  **Performance**: Are we doing unnecessary copies? Are we synchronizing device/host too often?
5.  **Style**: Does it match the project style (snake_case functions, PascalCase classes)?

## Specific Checks
-   **Headers**: Is `#pragma once` present?
-   **Namespaces**: Is everything in `vesper`?
-   **Tests**: Did the author add a test for the new feature?
