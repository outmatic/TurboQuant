# TurboQuant

**Near-optimal vector quantization for .NET 8+**

Compress embeddings to 2-4 bits with <0.5% quality loss. Zero dependencies. AOT-ready.

[![NuGet](https://img.shields.io/nuget/v/TurboQuant?style=flat-square&color=004880)](https://www.nuget.org/packages/TurboQuant)
[![Build](https://img.shields.io/github/actions/workflow/status/outmatic/TurboQuant/ci.yml?style=flat-square)](https://github.com/outmatic/TurboQuant/actions)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
![.NET](https://img.shields.io/badge/.NET-8%20%7C%2010-512bd4?style=flat-square)
![AOT](https://img.shields.io/badge/AOT-compatible-brightgreen?style=flat-square)

---

Implements **Algorithm 1** from [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026). Lloyd-Max quantization on the exact Beta(d/2, 1/2) distribution with randomized rotation.

```
768-dim float32 embedding  ──>  TurboQuant 4-bit  ──>  512 bytes
       3072 bytes                                       6x smaller
                                                        cosine > 0.995
```

## Install

```bash
dotnet add package TurboQuant
```

## 30-second Example

```csharp
using TurboQuant;

var quantizer = TurboQuantBuilder.Create(dim: 768).WithBits(4).BuildMSE();

var packed = quantizer.Quantize(embedding);        // float[768] -> PackedVector
var restored = quantizer.Dequantize(packed);       // PackedVector -> float[768]
var score = quantizer.ApproxSimilarity(a, b);      // LUT-based, zero allocation

// Zero-alloc dequantize into existing buffer
Span<float> buffer = stackalloc float[768];
quantizer.Dequantize(packed, buffer);

// Serialize for storage
var bytes = packed.ToBytes();
var loaded = PackedVector.FromBytes(bytes);
```

## Why TurboQuant?

|  | TurboQuant | Product Quantization | Scalar Quantization |
|--|-----------|---------------------|---------------------|
| **Quality (4-bit)** | cosine > 0.995 | cosine ~ 0.96 | cosine ~ 0.98 |
| **Quantization speed** | ~44 us/vec | ~500 ms/vec | ~1 us/vec |
| **Similarity search** | LUT, zero alloc | Asymmetric dist | Full decompress |
| **Needs training data?** | No | Yes (codebook training) | No |
| **Online / streaming?** | Yes | No (batch only) | Yes |
| **Theory-backed MSE?** | Yes (proven optimal) | No | No |

## Performance

Benchmarked on Apple Silicon (ARM64), .NET 10, Release mode:

| Operation | dim=256 | dim=768 | Heap Alloc |
|-----------|---------|---------|------------|
| `Quantize` | 8 us | 57 us | 152 B |
| `Dequantize` | 42 us | 42 us | 1 KB |
| `ApproxSimilarity` | 0.2 us | 0.8 us | **0 B** |
| `DotProduct` (SIMD) | 20 ns | 66 ns | **0 B** |
| `L2Norm` (SIMD) | 13 ns | 51 ns | **0 B** |
| `CosineSimilarity` (SIMD) | 35 ns | 126 ns | **0 B** |

SIMD operations use `Vector<T>` for automatic ISA selection (NEON on ARM, SSE/AVX2/AVX-512 on x86) with dual accumulators to hide data-dependency latency.

`ApproxSimilarity` uses a precomputed 16x16 lookup table for 4-bit. No unpacking, no centroid lookup. Direct byte-level scan.

`QuantizeBatch` processes vectors sequentially with zero overhead beyond the per-vector allocation.

## Compression Quality

Validated against the paper's theoretical bounds:

| Bits | Ratio | D_mse | Cosine | Paper Upper Bound |
|------|-------|-------|--------|-------------------|
| 4-bit | ~6-8x | <= 0.0095 | > 0.995 | 0.011 |
| 3-bit | ~8-10x | <= 0.035 | > 0.98 | 0.043 |
| 2-bit | ~12-16x | <= 0.118 | > 0.93 | 0.170 |

Ratios depend on whether the dimension is a power of 2 (no padding needed) or not.

---

## Use Cases

### Embedding Compression (OpenAI, Ollama, Azure AI)

```csharp
var embedding = await embeddingModel.GenerateAsync(["your text"]);
var vector = embedding[0].Vector.ToArray();

var quantizer = TurboQuantBuilder.Create(dim: 768).WithBits(4).BuildMSE();
var packed = quantizer.Quantize(vector);

// Store packed vectors in your DB, cache, or memory
// Search directly on compressed data (zero allocation):
var score = quantizer.ApproxSimilarity(queryPacked, docPacked);
```

**100K embeddings (768d): 292 MB -> 37 MB**

### RAG Pipeline: Compressed Vector Store

```csharp
var quantizer = TurboQuantBuilder.Create(dim: 768).WithBits(4).BuildMSE();
var store = new Dictionary<string, PackedVector>();

// Index
foreach (var doc in documents)
{
    var emb = await generator.GenerateAsync([doc.Text]);
    store[doc.Id] = quantizer.Quantize(emb[0].Vector.ToArray());
}

// Search (LUT-based, zero allocation)
var queryPacked = quantizer.Quantize(queryEmbedding);
var topK = store
    .Select(kv => (kv.Key, Score: quantizer.ApproxSimilarity(queryPacked, kv.Value)))
    .OrderByDescending(x => x.Score)
    .Take(10);
```

### LLM KV Cache: Extend Context Length

```csharp
var (cache, window) = TurboQuantBuilder
    .Create(dim: 128)             // head dimension
    .WithKeyBits(4)               // keys: 4-bit (attention precision)
    .WithValueBits(2)             // values: 2-bit (more aggressive)
    .WithResidualWindow(tokens: 128)
    .BuildKVCache();

// Append during generation. Recent tokens stay in float32,
// older tokens are automatically compressed
window.Append(keyVector, valueVector);

// Retrieve for attention computation
var allKeys = window.GetAllKeys();     // recent: exact, older: decompressed
var allValues = window.GetAllValues();
```

**32 layers x 32 heads x 4K tokens x 128d: 32 GB -> ~6 GB**

### Semantic Kernel: Compressed Memory

```csharp
public class CompressedMemoryStore : IMemoryStore
{
    private readonly TurboQuantMSE _quantizer;
    private readonly ConcurrentDictionary<string,
        ConcurrentDictionary<string, (PackedVector Vec, MemoryRecord Rec)>> _collections = new();

    public CompressedMemoryStore(int dim) =>
        _quantizer = TurboQuantBuilder.Create(dim).WithBits(4).BuildMSE();

    public Task<string> UpsertAsync(string collection, MemoryRecord record, CancellationToken ct)
    {
        var store = _collections.GetOrAdd(collection, _ => new());
        var packed = _quantizer.Quantize(record.Embedding.Vector.ToArray());
        store[record.Metadata.Id] = (packed, record);
        return Task.FromResult(record.Metadata.Id);
    }

    public Task<MemoryRecord?> GetNearestMatchAsync(
        string collection, ReadOnlyMemory<float> query, double minScore, CancellationToken ct)
    {
        if (!_collections.TryGetValue(collection, out var store))
            return Task.FromResult<MemoryRecord?>(null);

        var qPacked = _quantizer.Quantize(query.Span);
        return Task.FromResult(store.Values
            .Select(d => (d.Rec, Score: (double)_quantizer.ApproxSimilarity(qPacked, d.Vec)))
            .Where(x => x.Score >= minScore)
            .MaxBy(x => x.Score).Rec);
    }
}
```

### ML.NET Pipeline

```csharp
var quantizer = TurboQuantBuilder.Create(dim: 384).WithBits(4).BuildMSE();

var pipeline = mlContext.Transforms.CustomMapping<Input, Output>(
    (input, output) =>
    {
        var packed = quantizer.Quantize(input.Features);
        output.Compressed = quantizer.Dequantize(packed);
    },
    contractName: "TurboQuant");
```

### ONNX Runtime: Post-Inference Compression

```csharp
using var session = new InferenceSession("embedding_model.onnx");
var quantizer = TurboQuantBuilder.Create(dim: 768).WithBits(4).BuildMSE();

using var results = session.Run(inputs);
var outputTensor = results.First().AsEnumerable<float>().ToArray();

var compressed = new PackedVector[batchSize];
quantizer.QuantizeBatch(outputTensor, compressed);
```

### Serialization

```csharp
// Save
var bytes = packed.ToBytes();
File.WriteAllBytes("vector.bin", bytes);

// Load
var loaded = PackedVector.FromBytes(File.ReadAllBytes("vector.bin"));
var restored = quantizer.Dequantize(loaded);
```

---

## Rotation Strategies

TurboQuant defaults to **random orthogonal rotation** (QR decomposition of a Gaussian matrix), which matches the paper exactly and works with any dimension without overhead.

For **power-of-2 dimensions** (128, 256, 512, 1024), the **Hadamard rotation** is an excellent alternative: identical quality, O(d log d) time, and O(d) memory instead of O(d^2).

For **non-power-of-2 dimensions** (384, 768, 1536), Hadamard pads internally to the next power of 2, which increases packed data size by up to 33%. A `Trace.TraceWarning` is emitted when this happens.

```csharp
// Default: random orthogonal (paper-correct, any dimension)
var quantizer = TurboQuantBuilder.Create(dim: 768).WithBits(4).BuildMSE();

// Fast path for power-of-2 dimensions (recommended for KV cache with head dim 64/128)
var fast = TurboQuantBuilder.Create(dim: 1024).WithBits(4).WithHadamardRotation().BuildMSE();
```

| | Random Orthogonal (default) | Hadamard |
|---|---|---|
| **Time** | O(d^2) | O(d log d) |
| **Memory** | O(d^2) per quantizer | O(d) per quantizer |
| **Packed size** | Exact (d coordinates) | Exact if power-of-2, padded otherwise |
| **dim=768 quantize** | ~57 us | ~1.3 us (but 33% more storage) |
| **dim=1024 quantize** | ~100 us | ~1.6 us (no padding) |
| **D_mse** | = theoretical | = theoretical (power-of-2), better (non-power-of-2) |

**Best practice:** if your embedding model supports configurable dimensions (e.g. OpenAI, nomic), choose a power-of-2 dimension (512, 1024) to get the best of both strategies.

> **Note on seed correlation:** the rotation matrix is generated from a random seed. If your test vectors are generated from the *same* seed (e.g. `new Random(42)` for both), the rotation and data become correlated, causing degraded quality. This does not happen in practice (embeddings come from a model, not from `Random`), but use distinct seeds in synthetic benchmarks.

## How It Works

```
                    ┌──────────────┐
   float[768]  ──>  │  Normalize   │ store ||x||
                    └──────┬───────┘
                           v
                    ┌──────────────┐
                    │   Rotate     │ random orthogonal (default) or Hadamard
                    └──────┬───────┘
                           v
                    ┌──────────────┐
                    │  Quantize    │ per-coordinate Lloyd-Max on Beta(d/2, 1/2)
                    └──────┬───────┘
                           v
                    ┌──────────────┐
                    │  Bit-Pack    │ 4-bit: 2 values/byte, 3-bit: 8 values/3 bytes
                    └──────┬───────┘
                           v
                    PackedVector (data + norm)
```

The rotation ensures each coordinate is approximately Beta-distributed regardless of input, enabling **provably near-optimal** scalar quantization. D_mse is within factor sqrt(3)pi/2 ~ 2.7x of the information-theoretic lower bound (Theorem 3 in the paper).

`ApproxSimilarity` computes cosine similarity directly on packed bytes via a precomputed 16x16 centroid dot-product lookup table (4-bit). No unpacking, no centroid lookup, zero heap allocation.

## Project Layout

```
src/TurboQuant/
  Core/
    Rotation/       RandomRotation (default, paper-correct), HadamardRotation (fast, power-of-2)
    Codebook/       BetaCodebook (Lloyd-Max on exact Beta distribution, with LUT)
    Packing/        BitPacker (2/3/4-bit), PackedVector (serializable, immutable)
    Quantizers/     TurboQuantMSE, TurboQuantAsymmetric
    Simd/           DotProductSimd (Vector<T> auto-ISA: NEON/SSE/AVX2/AVX-512)
  Cache/            KVCache, ResidualWindow
  Diagnostics/      CompressionStats (thread-safe), QuantizationBenchmark
  TurboQuantBuilder.cs
```

## Build & Test

```bash
dotnet build TurboQuantNet.slnx
dotnet test
dotnet run --project samples/TurboQuant.Demo
dotnet run -c Release --project benchmarks/TurboQuant.Benchmarks
```

## License

[MIT](LICENSE). Based on the paper by Zandieh, Daliri, Hadian & Mirrokni (Google Research / NYU / Google DeepMind).
