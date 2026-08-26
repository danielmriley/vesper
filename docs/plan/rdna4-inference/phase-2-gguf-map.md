# Phase 2. GGUF map

Back to [overview](overview.md).

## Goal

Vesper can open a GGUF file, list tensor names, types, and shapes,
and mmap the payload. No dequant. No generate. You need a real file
layout before a Q8 kernel has something to read.

## Changes

`include/vesper/gguf.h` and `src/gguf.cpp` parse the header and the
tensor table. A small test writes a fixture with one or two tensors
and checks names and byte offsets.

Do not touch `Engine` yet.

## Data structures

`GgufFile` owns the mmap. `GgufTensor` is name, type tag, shape, and
a pointer into the map. Type tags you accept now are F32, F16, and
Q8_0. Reject other types with a clear error.

## Verification

**Static.** `ctest --test-dir build --output-on-failure`

**Runtime.** No CLI flag in this phase. The fixture test is the
check. There is no control skill for a mmap parser. Add
`--gguf-info FILE` only if the test cannot show the table.
