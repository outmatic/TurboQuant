# Changelog

## 0.1.0

Initial release implementing the TurboQuant algorithm from "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh et al., ICLR 2026).

### Features
- Paper-correct random orthogonal rotation (default) via QR decomposition
- Optional Hadamard rotation for O(d log d) fast path on power-of-2 dimensions
- Lloyd-Max codebook optimized for the exact Beta(d/2, 1/2) distribution
- 2/3/4-bit per-coordinate scalar quantization
- Real bit-packing (not simulated) with zero-allocation Span API
- LUT-based approximate similarity (4-bit): zero allocation, direct byte scan
- PackedVector serialization (ToBytes/FromBytes)
- Immutable PackedVector (ReadOnlyMemory)
- Zero-allocation Dequantize via Span overload
- Thread-safe CompressionStats (atomic double via CAS)
- Asymmetric KV cache compression (separate key/value bit widths)
- Residual window for hybrid compressed/uncompressed KV cache
- SIMD-accelerated operations (AVX2/SSE2/NEON via .NET intrinsics)
- Fluent builder API with Trace.TraceWarning for non-power-of-2 Hadamard
- AOT and trimming compatible
- Multi-target: .NET 8 and .NET 10
- Zero external dependencies

### Validation
- D_mse matches theoretical Lloyd-Max distortion within 1%
- Satisfies paper upper bound D_mse <= sqrt(3)*pi/2 * 4^(-b)
- 4-bit cosine similarity > 0.995 on dim=768
- 125 tests covering paper validation, end-to-end scenarios, robustness, and thread safety
