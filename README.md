# dmat4 storage-buffer load from a 16-byte-aligned base

Self-contained Vulkan compute program that loads a `dmat4` from a storage buffer
bound at a base address that is 16-byte but not 32-byte aligned and compares the
result with the buffer contents.

## Build and run

Requires CMake 3.19+, a C++17 compiler and the Vulkan SDK (for `vulkan.h`, the
loader and `glslc` or `glslangValidator`).

```
run.cmd            # Windows
./run.sh           # Linux, macOS
```

Both scripts configure, build and run; arguments are passed to the program
(`run.cmd --validate`). Manual equivalent:

```
cmake -B build
cmake --build build --config Release
build/Release/dmat4_repro.exe        # Windows
build/dmat4_repro                    # Linux
```

Options: `--list` prints all physical devices; `--gpu N` selects one (default:
the first with `shaderFloat64`); `--host` puts the input buffers in host-visible
instead of device-local memory; `--validate` enables the Khronos validation layer
and prints its messages (none are expected).

The exit code is 0 when every case passes, 1 otherwise, 2 without `shaderFloat64`.

## What it does

The input buffer holds 8 padding doubles (-1 .. -8) followed by 128 doubles
3k + 0.25. The shader (`copy.comp`) loads `dmat4 m = matrices[i]` for i in 0..7
and stores the 16 components of `m` as doubles, so the output must equal the
bound range verbatim.

Section A binds the buffer (bound to memory at offset 0) with descriptor offsets
0, 8, 16, 24, 32, 48 and 64 bytes; offsets that are not a multiple of
`minStorageBufferOffsetAlignment` are skipped. Section B binds copies of the
buffer to memory at offsets 16, 32 and 48 bytes (skipped if not a multiple of
`VkMemoryRequirements::alignment`) with descriptor offset 0. In both sections
the bases at 16 and 48 are 16-byte aligned but not 32-byte aligned.

A failing case prints the first mismatch, the first values, and whether the
output equals the buffer contents read from a shifted address.

## Expected result

All cases pass: every offset used is a multiple of the alignment the driver
itself reports (`minStorageBufferOffsetAlignment` for descriptor offsets,
`VkMemoryRequirements::alignment` for memory offsets), and the Vulkan
specification imposes no further alignment on the start of a storage buffer
range. The std430 base alignment of `dvec4` (32 bytes) constrains member offsets
within the block, not the block's base.

## Observed results

The program ends with a line `results row (paste into README.md):` followed by
one markdown row in the format of this table. The last four columns are the
cases with a 16 mod 32 base: `A16`/`A48` are descriptor offsets 16 and 48 on a
buffer bound at memory offset 0, `B16`/`B48` are memory offsets 16 and 48 with
descriptor offset 0. A failure names the shift the output shows.

| GPU | OS | driver | min offset align | req. alignment | A16 | A48 | B16 | B48 |
|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 5070 Ti | Windows 10.0 build 26100 | NVIDIA 621.40 | 16 | 16 | FAIL (16 B early) | FAIL (16 B early) | FAIL (16 B early) | FAIL (16 B early) |
| NVIDIA GeForce RTX 4070 Ti SUPER | Windows 10.0 build 26100 | NVIDIA 610.88 | 16 | 16 | pass | pass | pass | pass |
| NVIDIA GeForce RTX 4070 Ti SUPER | Windows 10.0 build 26100 | NVIDIA 621.01 | 16 | 16 | pass | pass | pass | pass |
| NVIDIA GeForce RTX 4070 Ti SUPER | Windows 10.0 build 26100 | NVIDIA 621.40 | 16 | 16 | pass | pass | pass | pass |
| AMD Radeon RX 9060 XT | Windows 10.0 build 26300 | AMD proprietary driver 26.11.2 (LLPC) | 4 | 4 | pass | pass | pass | pass |
| AMD Radeon Graphics (RADV GFX1200) | Linux 7.0.0-43-generic | radv Mesa 25.2.8-0ubuntu0.24.04.5 | 4 | 16 | pass | pass | pass | pass |
