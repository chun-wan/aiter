# gfx90a BF16 GEMV Kernels for Gemma 4 31B Decode

## Target: AMD MI250/MI250X (CDNA2, gfx90a)

## Key Shapes (Gemma 4 31B, TP=2)

| Operation | M | N | K | Bound | Current Perf |
|-----------|---|---|---|-------|-------------|
| QKV proj decode | 1 | 4096 | 5376 | Memory | 0.86 TFLOPS (0.9% peak) |
| FFN up decode | 1 | 10752 | 5376 | Memory | 1.29 TFLOPS (1.3% peak) |
| FFN down decode | 1 | 5376 | 10752 | Memory | 1.10 TFLOPS (1.1% peak) |
| Global KV proj | 1 | 2048 | 5376 | Memory | 0.49 TFLOPS |

## CDNA2 ISA Notes (vs CDNA3/gfx942)

| Feature | gfx90a (CDNA2) | gfx942 (CDNA3) |
|---------|---------------|----------------|
| BF16 MFMA | v_mfma_f32_32x32x8bf16_1k | v_mfma_f32_32x32x16bf16 |
| BF16 Dot | v_dot2c_f32_bf16 | v_dot2c_f32_bf16 |
| VGPRs/wave | 256 + 256 AGPR | 256 + 256 AGPR |
| LDS | 64KB, 32 banks | 64KB, 32 banks |
| HBM BW | 1.6 TB/s per GCD | 5.3 TB/s per GCD |
| CUs/GCD | 104 | 228 |
| FP8 MFMA | N/A | Yes |

## GEMV Optimization Strategy

For M=1 (decode), the kernel is **purely memory-bound**:
- Arithmetic intensity: ~1 FLOP/byte (well below ridge point of 60)
- Target: maximize HBM bandwidth utilization (>80% of 1.6 TB/s = 1.28 TB/s)
- Current hipBLASLt achieves ~54-80% BW utilization (0.86-1.29 TB/s)

### Approach
1. **Vectorized loads**: buffer_load_dwordx4 (128 bits = 8 BF16 per instruction)
2. **BF16 dot accumulation**: v_dot2c_f32_bf16 (2 BF16 FMA per clock)
3. **K-splitting**: Distribute K across workgroups, LDS reduction for partial sums
4. **Output tiling**: Each workgroup computes a contiguous tile of N outputs
5. **Prefetching**: Double-buffer with async global loads

### Build
```bash
./build_gemv.sh  # Requires ROCm with LLVM/clang
```

## Status: SCAFFOLD
The CSV dispatch config and build infrastructure are in place.
Actual ASM kernel implementation requires:
1. CDNA2 ISA micro-benchmarking for optimal instruction scheduling
2. Memory access pattern tuning for MI250 HBM2e controller
3. LDS bank conflict analysis for K-reduction
4. Comparison vs hipBLASLt baseline for each shape
