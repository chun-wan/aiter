#!/bin/bash
# Build gfx90a BF16 GEMV kernels from assembly source
# Usage: ./build_gemv.sh
#
# Prerequisites: ROCm with hipcc/clang, target gfx90a
#
# Architecture: AMD CDNA2 (MI250/MI250X)
# Key ISA differences from CDNA3 (gfx942):
#   - v_mfma_f32_32x32x8bf16_1k (vs gfx942's v_mfma_f32_32x32x16bf16)
#   - v_dot2c_f32_bf16 available for 2-element BF16 dot product
#   - 256 VGPRs + 256 AGPRs per wave
#   - 64KB LDS, 32 banks
#   - 104 CUs per GCD
#
# GEMV Strategy (M=1, memory-bound):
#   - Maximize HBM bandwidth utilization (target >80% of 1.6 TB/s)
#   - Use buffer_load_dwordx4 for 128-bit vector loads (8 BF16 values)
#   - Split K across workgroups, reduce via LDS or global atomics
#   - Each workgroup handles a tile of N output elements
#
# NOTE: This is a development scaffold. Full ASM kernel implementation
# requires the AMD CDNA2 ISA reference and extensive micro-benchmarking.
# The actual .co files need to be compiled from .s assembly using:
#   /opt/rocm/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx90a \
#     -mcode-object-version=4 -o kernel.co kernel.s

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLANG="/opt/rocm/llvm/bin/clang"
TARGET="gfx90a"

echo "=== Building gfx90a BF16 GEMV kernels ==="
echo "Target: $TARGET (CDNA2, MI250)"

if [ ! -x "$CLANG" ]; then
    echo "ERROR: $CLANG not found. Need ROCm installation."
    exit 1
fi

# For each .s file in this directory, compile to .co
for src in "$SCRIPT_DIR"/*.s; do
    [ -f "$src" ] || continue
    base=$(basename "$src" .s)
    out="$SCRIPT_DIR/${base}.co"
    echo "  Compiling: $base.s -> $base.co"
    $CLANG -target amdgcn-amd-amdhsa -mcpu=$TARGET \
        -mcode-object-version=4 \
        -o "$out" "$src" 2>&1
done

echo "=== Build complete ==="
ls -la "$SCRIPT_DIR"/*.co 2>/dev/null || echo "No .co files generated (need .s source files)"
