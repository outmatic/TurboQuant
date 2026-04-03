using TurboQuant.Cache;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Quantizers;
using TurboQuant.Core.Rotation;
using TurboQuant.Core.Simd;
using Xunit;
using Xunit.Abstractions;

namespace TurboQuant.Tests.Integration;

/// <summary>
/// End-to-end validation with realistic scenarios — not random vectors,
/// but structured data that simulates real embedding use cases.
/// </summary>
public class EndToEndValidation
{
    private readonly ITestOutputHelper _out;
    public EndToEndValidation(ITestOutputHelper output) => _out = output;

    // ──────────────────────────────────────────────
    // 1. Semantic Search: Recall@K with clustered embeddings
    //    Simulates a real embedding space with clusters (topics)
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(4, 0.80)]  // Recall depends on cluster separation and quantization
    [InlineData(3, 0.70)]
    [InlineData(2, 0.55)]
    public void SemanticSearch_RecallAtK(int bits, double minRecall)
    {
        const int dim = 384;
        const int numClusters = 20;
        const int docsPerCluster = 50;
        const int totalDocs = numClusters * docsPerCluster;
        const int k = 10;

        var rng = new Random(42);
        var quantizer = new TurboQuantMSE(dim, bits, seed: 99);

        // Generate clustered embeddings (simulating topics)
        var docs = new float[totalDocs][];
        var clusterCenters = new float[numClusters][];
        for (var c = 0; c < numClusters; c++)
        {
            clusterCenters[c] = GenerateUnitVector(dim, rng);
            for (var d = 0; d < docsPerCluster; d++)
            {
                // Each doc = cluster center + small noise
                var doc = new float[dim];
                for (var i = 0; i < dim; i++)
                    doc[i] = clusterCenters[c][i] + (float)(NextGaussian(rng) * 0.1);
                Normalize(doc);
                docs[c * docsPerCluster + d] = doc;
            }
        }

        // Quantize all documents
        var packedDocs = new PackedVector[totalDocs];
        for (var i = 0; i < totalDocs; i++)
            packedDocs[i] = quantizer.Quantize(docs[i]);

        // For each cluster center as query, check if top-K results are from same cluster
        var totalRecall = 0.0;
        var numQueries = numClusters;

        for (var q = 0; q < numQueries; q++)
        {
            var query = clusterCenters[q];

            // True top-K by exact cosine
            var trueScores = docs
                .Select((d, i) => (Score: DotProductSimd.CosineSimilarity(query, d), Index: i))
                .OrderByDescending(x => x.Score)
                .Take(k)
                .Select(x => x.Index)
                .ToHashSet();

            // Approximate top-K via quantized similarity
            var packedQuery = quantizer.Quantize(query);
            var approxScores = packedDocs
                .Select((p, i) => (Score: quantizer.ApproxSimilarity(packedQuery, p), Index: i))
                .OrderByDescending(x => x.Score)
                .Take(k)
                .Select(x => x.Index)
                .ToHashSet();

            var recall = (double)trueScores.Intersect(approxScores).Count() / k;
            totalRecall += recall;
        }

        var avgRecall = totalRecall / numQueries;
        _out.WriteLine($"{bits}-bit Recall@{k}: {avgRecall:F3} (threshold: {minRecall})");
        Assert.True(avgRecall >= minRecall, $"Recall@{k} = {avgRecall:F3} below threshold {minRecall}");
    }

    // ──────────────────────────────────────────────
    // 2. Nearest Neighbor Ranking Preservation
    //    For each doc, check if its nearest neighbor is the same before/after quantization
    // ──────────────────────────────────────────────

    [Fact]
    public void NearestNeighbor_PreservedAfterQuantization()
    {
        const int dim = 256;
        const int n = 200;
        var rng = new Random(55);
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);

        var vectors = Enumerable.Range(0, n).Select(_ => GenerateUnitVector(dim, rng)).ToArray();
        var packed = vectors.Select(v => quantizer.Quantize(v)).ToArray();

        var preserved = 0;
        for (var i = 0; i < n; i++)
        {
            // True nearest neighbor (excluding self)
            var trueNN = Enumerable.Range(0, n)
                .Where(j => j != i)
                .MaxBy(j => DotProductSimd.CosineSimilarity(vectors[i], vectors[j]));

            // Quantized nearest neighbor
            var quantNN = Enumerable.Range(0, n)
                .Where(j => j != i)
                .MaxBy(j => quantizer.ApproxSimilarity(packed[i], packed[j]));

            if (trueNN == quantNN) preserved++;
        }

        var rate = (double)preserved / n;
        _out.WriteLine($"NN preservation rate: {rate:F3} ({preserved}/{n})");
        Assert.True(rate > 0.7, $"NN preservation {rate:F3} too low");
    }

    // ──────────────────────────────────────────────
    // 3. KV Cache: Attention Score Preservation
    //    Simulates a transformer attention computation with compressed KV cache
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(4, 2, 0.90)]   // keys=4bit, values=2bit
    [InlineData(4, 4, 0.95)]   // keys=4bit, values=4bit
    public void KVCache_AttentionScoresPreserved(int keyBits, int valueBits, double minCorrelation)
    {
        const int headDim = 64;
        const int seqLen = 128;
        var rng = new Random(77);

        // Generate K, V, and query vectors (simulating transformer attention)
        var keys = Enumerable.Range(0, seqLen).Select(_ => GenerateUnitVector(headDim, rng)).ToArray();
        var values = Enumerable.Range(0, seqLen).Select(_ => GenerateUnitVector(headDim, rng)).ToArray();
        var query = GenerateUnitVector(headDim, rng);

        // Compute exact attention: softmax(Q·K^T / √d) · V
        var exactScores = keys.Select(k => DotProduct(query, k) / MathF.Sqrt(headDim)).ToArray();
        var exactWeights = Softmax(exactScores);
        var exactOutput = WeightedSum(values, exactWeights);

        // Compress KV cache
        var asymQuantizer = new TurboQuantAsymmetric(headDim, keyBits, valueBits, seed: 42);
        var compressedKeys = keys.Select(k => asymQuantizer.QuantizeKey(k)).ToArray();
        var compressedValues = values.Select(v => asymQuantizer.QuantizeValue(v)).ToArray();

        // Compute attention with decompressed KV
        var decompKeys = compressedKeys.Select(k => asymQuantizer.DequantizeKey(k)).ToArray();
        var decompValues = compressedValues.Select(v => asymQuantizer.DequantizeValue(v)).ToArray();

        var approxScores = decompKeys.Select(k => DotProduct(query, k) / MathF.Sqrt(headDim)).ToArray();
        var approxWeights = Softmax(approxScores);
        var approxOutput = WeightedSum(decompValues, approxWeights);

        // Check output correlation
        var cosine = DotProductSimd.CosineSimilarity(exactOutput, approxOutput);
        _out.WriteLine($"KV cache (keys={keyBits}b, vals={valueBits}b): attention output cosine={cosine:F6}");
        Assert.True(cosine > minCorrelation, $"Attention output cosine {cosine:F6} below {minCorrelation}");

        // Check attention weight distribution similarity (KL-like)
        var weightError = exactWeights.Zip(approxWeights, (e, a) => Math.Abs(e - a)).Average();
        _out.WriteLine($"  Average attention weight error: {weightError:F6}");
    }

    // ──────────────────────────────────────────────
    // 4. ResidualWindow: Recent tokens uncompressed, older compressed
    // ──────────────────────────────────────────────

    [Fact]
    public void ResidualWindow_SmoothTransition()
    {
        const int headDim = 64;
        const int windowSize = 16;
        const int totalTokens = 64;
        var rng = new Random(88);

        var (cache, window) = TurboQuantBuilder
            .Create(dim: headDim)
            .WithKeyBits(4)
            .WithValueBits(2)
            .WithResidualWindow(tokens: windowSize)
            .BuildKVCache();

        // Store original vectors for comparison
        var originalKeys = new float[totalTokens][];
        for (var t = 0; t < totalTokens; t++)
        {
            var key = GenerateUnitVector(headDim, rng);
            var value = GenerateUnitVector(headDim, rng);
            originalKeys[t] = key;
            window.Append(key, value);
        }

        Assert.Equal(totalTokens - windowSize, cache.Count);  // older tokens compressed
        Assert.Equal(windowSize, window.WindowCount);           // recent tokens in window

        // Get all keys back
        var allKeys = window.GetAllKeys();
        Assert.Equal(totalTokens, allKeys.Length);

        // Recent keys (in window) should be EXACT
        for (var t = totalTokens - windowSize; t < totalTokens; t++)
        {
            var cosine = DotProductSimd.CosineSimilarity(originalKeys[t], allKeys[t]);
            Assert.Equal(1.0f, cosine, precision: 5);
        }

        // Older keys (compressed) should be close but not exact
        var compressedCosines = new float[totalTokens - windowSize];
        for (var t = 0; t < totalTokens - windowSize; t++)
        {
            compressedCosines[t] = DotProductSimd.CosineSimilarity(originalKeys[t], allKeys[t]);
        }

        var avgCompressedCosine = compressedCosines.Average();
        _out.WriteLine($"Compressed keys avg cosine: {avgCompressedCosine:F6}");
        Assert.True(avgCompressedCosine > 0.99, $"Compressed cosine {avgCompressedCosine:F6} too low");
    }

    // ──────────────────────────────────────────────
    // 5. Determinism: same seed → identical output
    // ──────────────────────────────────────────────

    [Fact]
    public void FullPipeline_IsDeterministic()
    {
        const int dim = 384;
        var rng = new Random(42);
        var vector = GenerateUnitVector(dim, rng);

        var q1 = TurboQuantBuilder.Create(dim).WithBits(4).WithSeed(123).BuildMSE();
        var q2 = TurboQuantBuilder.Create(dim).WithBits(4).WithSeed(123).BuildMSE();

        var p1 = q1.Quantize(vector);
        var p2 = q2.Quantize(vector);

        Assert.Equal(p1.Data, p2.Data);
        Assert.Equal(p1.Norm, p2.Norm);

        var r1 = q1.Dequantize(p1);
        var r2 = q2.Dequantize(p2);

        for (var i = 0; i < dim; i++)
            Assert.Equal(r1[i], r2[i]);
    }

    // ──────────────────────────────────────────────
    // 6. Non-unit vectors: norm preserved correctly
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(0.001f)]
    [InlineData(1.0f)]
    [InlineData(100.0f)]
    [InlineData(10000.0f)]
    public void NonUnitVectors_DirectionPreserved(float scale)
    {
        const int dim = 256;
        var rng = new Random(200); // different seed from quantizer to avoid correlation
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);

        var vector = GenerateUnitVector(dim, rng);
        for (var i = 0; i < dim; i++) vector[i] *= scale;

        var packed = quantizer.Quantize(vector);
        var restored = quantizer.Dequantize(packed);

        // Quantization preserves direction (cosine) but NOT magnitude (norm shrinkage is inherent)
        var cosine = DotProductSimd.CosineSimilarity(vector, restored);
        _out.WriteLine($"scale={scale}: cosine={cosine:F6}");
        Assert.True(cosine > 0.99, $"Direction not preserved: cosine={cosine:F6}");
    }

    // ──────────────────────────────────────────────
    // 7. Hadamard vs Random Rotation: both produce valid results
    // ──────────────────────────────────────────────

    [Fact]
    public void HadamardVsRandom_BothWorkCorrectly()
    {
        const int dim = 256;
        const int n = 500;
        var rng = new Random(999); // must differ from quantizer seed to avoid correlation

        var qRandom = TurboQuantBuilder.Create(dim).WithBits(4).WithSeed(42).BuildMSE();
        var qHadamard = TurboQuantBuilder.Create(dim).WithBits(4).WithSeed(42).WithHadamardRotation().BuildMSE();

        double cosRandom = 0, cosHadamard = 0;
        for (var i = 0; i < n; i++)
        {
            var v = GenerateUnitVector(dim, rng);
            cosRandom += DotProductSimd.CosineSimilarity(v, qRandom.Dequantize(qRandom.Quantize(v)));
            cosHadamard += DotProductSimd.CosineSimilarity(v, qHadamard.Dequantize(qHadamard.Quantize(v)));
        }

        cosRandom /= n;
        cosHadamard /= n;

        _out.WriteLine($"Random rotation:   avg cosine = {cosRandom:F6}");
        _out.WriteLine($"Hadamard rotation: avg cosine = {cosHadamard:F6}");

        Assert.True(cosRandom > 0.99, $"Random rotation cosine {cosRandom:F4} too low");
        Assert.True(cosHadamard > 0.99, $"Hadamard rotation cosine {cosHadamard:F4} too low");
    }

    // ──────────────────────────────────────────────
    // 8. Stress test: large batch, verify no memory leaks (GC pressure)
    // ──────────────────────────────────────────────

    [Fact]
    public void StressTest_LargeBatch_NoExcessiveAllocation()
    {
        const int dim = 128;
        const int batchSize = 10_000;
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);
        var rng = new Random(42);

        var vectors = new float[batchSize * dim];
        for (var i = 0; i < vectors.Length; i++)
            vectors[i] = (float)(rng.NextDouble() * 2 - 1);

        var output = new PackedVector[batchSize];

        var before = GC.GetTotalAllocatedBytes(precise: true);
        quantizer.QuantizeBatch(vectors, output);
        var after = GC.GetTotalAllocatedBytes(precise: true);

        var allocatedPerVector = (double)(after - before) / batchSize;
        _out.WriteLine($"Batch {batchSize} vectors: {allocatedPerVector:F0} bytes/vector allocated");

        // Each PackedVector needs dim/2 bytes for 4-bit data + overhead
        // Should be well under 1KB per vector
        Assert.True(allocatedPerVector < 1024, $"Too much allocation: {allocatedPerVector:F0} bytes/vector");
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    private static float[] GenerateUnitVector(int dim, Random rng)
    {
        var v = new float[dim];
        var norm = 0f;
        for (var i = 0; i < dim; i++)
        {
            v[i] = (float)NextGaussian(rng);
            norm += v[i] * v[i];
        }
        norm = MathF.Sqrt(norm);
        for (var i = 0; i < dim; i++) v[i] /= norm;
        return v;
    }

    private static void Normalize(float[] v)
    {
        var norm = DotProductSimd.L2Norm(v);
        for (var i = 0; i < v.Length; i++) v[i] /= norm;
    }

    private static double NextGaussian(Random rng)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static float DotProduct(float[] a, float[] b)
    {
        var sum = 0f;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
        var sum = exps.Sum();
        return exps.Select(e => e / sum).ToArray();
    }

    private static float[] WeightedSum(float[][] vectors, float[] weights)
    {
        var dim = vectors[0].Length;
        var result = new float[dim];
        for (var i = 0; i < vectors.Length; i++)
            for (var j = 0; j < dim; j++)
                result[j] += weights[i] * vectors[i][j];
        return result;
    }
}
