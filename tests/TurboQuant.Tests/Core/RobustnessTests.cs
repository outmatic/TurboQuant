using TurboQuant.Core.Codebook;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Quantizers;
using TurboQuant.Core.Simd;
using Xunit;
using Xunit.Abstractions;

namespace TurboQuant.Tests.Core;

public class RobustnessTests
{
    private readonly ITestOutputHelper _out;
    public RobustnessTests(ITestOutputHelper output) => _out = output;

    // ──────────────────────────────────────────────
    // 1. Thread Safety
    // ──────────────────────────────────────────────

    [Fact]
    public void CompressionStats_ThreadSafe_UnderContention()
    {
        var quantizer = new TurboQuantMSE(128, 4, seed: 42);
        var rng = new Random(42);
        var vectors = Enumerable.Range(0, 100)
            .Select(_ => RandomUnitVector(128, rng))
            .ToArray();

        // Hammer from 8 threads concurrently
        Parallel.For(0, 1000, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
        {
            quantizer.Quantize(vectors[i % vectors.Length]);
        });

        var stats = quantizer.GetStats();
        Assert.Equal(1000, stats.TotalVectorsProcessed);
        Assert.True(stats.AverageMSE > 0);
        _out.WriteLine($"Processed {stats.TotalVectorsProcessed} vectors across 8 threads, avg D_mse={stats.AverageMSE:F6}");
    }

    [Fact]
    public void Quantizer_ConcurrentQuantize_ProducesValidResults()
    {
        var quantizer = new TurboQuantMSE(256, 4, seed: 42);
        var rng = new Random(99);
        var vectors = Enumerable.Range(0, 50)
            .Select(_ => RandomUnitVector(256, rng))
            .ToArray();

        var results = new PackedVector[vectors.Length];

        Parallel.For(0, vectors.Length, i =>
        {
            results[i] = quantizer.Quantize(vectors[i]);
        });

        // All results should be valid and produce high cosine on dequantize
        for (var i = 0; i < vectors.Length; i++)
        {
            var restored = quantizer.Dequantize(results[i]);
            var cosine = DotProductSimd.CosineSimilarity(vectors[i], restored);
            Assert.True(cosine > 0.98, $"Vector {i}: cosine {cosine:F4} too low after concurrent quantize");
        }
    }

    // ──────────────────────────────────────────────
    // 2. BitPacker Fuzz / Malformed Input
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void BitPacker_IndicesClampedToValidRange(int bits)
    {
        var levels = 1 << bits;
        var indices = new int[] { 0, levels - 1, 0, levels - 1, 0, levels - 1, 0, levels - 1 };

        var packedBytes = PackedVector.GetPackedByteCount(indices.Length, bits);
        var packed = new byte[packedBytes];
        BitPacker.Pack(indices, packed, bits);

        var unpacked = new int[indices.Length];
        BitPacker.Unpack(packed, unpacked, indices.Length, bits);

        for (var i = 0; i < indices.Length; i++)
            Assert.Equal(indices[i], unpacked[i]);
    }

    [Theory]
    [InlineData(2, 1)]
    [InlineData(3, 1)]
    [InlineData(4, 1)]
    [InlineData(2, 3)]
    [InlineData(3, 5)]
    [InlineData(4, 7)]
    public void BitPacker_OddCounts(int bits, int count)
    {
        var rng = new Random(42);
        var levels = 1 << bits;
        var indices = Enumerable.Range(0, count).Select(_ => rng.Next(levels)).ToArray();

        var packedBytes = PackedVector.GetPackedByteCount(count, bits);
        var packed = new byte[packedBytes];
        BitPacker.Pack(indices, packed, bits);

        var unpacked = new int[count];
        BitPacker.Unpack(packed, unpacked, count, bits);

        for (var i = 0; i < count; i++)
            Assert.Equal(indices[i], unpacked[i]);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void BitPacker_LargeRandomFuzz(int bits)
    {
        var rng = new Random(777);
        var levels = 1 << bits;

        for (var trial = 0; trial < 100; trial++)
        {
            var count = rng.Next(1, 2048);
            var indices = Enumerable.Range(0, count).Select(_ => rng.Next(levels)).ToArray();

            var packedBytes = PackedVector.GetPackedByteCount(count, bits);
            var packed = new byte[packedBytes];
            BitPacker.Pack(indices, packed, bits);

            var unpacked = new int[count];
            BitPacker.Unpack(packed, unpacked, count, bits);

            for (var i = 0; i < count; i++)
                Assert.Equal(indices[i], unpacked[i]);
        }
    }

    [Fact]
    public void BitPacker_InvalidBits_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => BitPacker.Pack(new int[8], new byte[8], 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => BitPacker.Unpack(new byte[8], new int[8], 8, 1));
    }

    // ──────────────────────────────────────────────
    // 3. Edge Case Dimensions
    // ──────────────────────────────────────────────

    [Fact]
    public void MinDimension_2_Works()
    {
        var quantizer = new TurboQuantMSE(2, 4, seed: 42);
        var v = new float[] { 0.6f, 0.8f }; // unit vector

        var packed = quantizer.Quantize(v);
        var restored = quantizer.Dequantize(packed);

        Assert.Equal(2, restored.Length);
        // At dim=2 quality is lower, but it should still work without crashing
        Assert.True(float.IsFinite(restored[0]));
        Assert.True(float.IsFinite(restored[1]));
    }

    [Fact]
    public void SmallDimension_8_ReasonableQuality()
    {
        var quantizer = new TurboQuantMSE(8, 4, seed: 42);
        var rng = new Random(200);

        double totalCosine = 0;
        for (var i = 0; i < 100; i++)
        {
            var v = RandomUnitVector(8, rng);
            totalCosine += DotProductSimd.CosineSimilarity(v, quantizer.Dequantize(quantizer.Quantize(v)));
        }

        var avg = totalCosine / 100;
        _out.WriteLine($"dim=8, 4-bit: avg cosine = {avg:F4}");
        Assert.True(avg > 0.85, $"dim=8 cosine {avg:F4} unexpectedly low");
    }

    [Fact]
    public void LargeDimension_4096_Works()
    {
        // Just verify it doesn't crash or OOM
        var quantizer = new TurboQuantMSE(4096, 4, seed: 42);
        var rng = new Random(999);
        var v = RandomUnitVector(4096, rng);

        var packed = quantizer.Quantize(v);
        var restored = quantizer.Dequantize(packed);

        Assert.Equal(4096, restored.Length);
        var cosine = DotProductSimd.CosineSimilarity(v, restored);
        _out.WriteLine($"dim=4096, 4-bit: cosine = {cosine:F6}");
        Assert.True(cosine > 0.99);
    }

    // ──────────────────────────────────────────────
    // 4. NaN / Inf / Degenerate Vectors
    // ──────────────────────────────────────────────

    [Fact]
    public void ZeroVector_ReturnsZero()
    {
        var quantizer = new TurboQuantMSE(64, 4, seed: 42);
        var zero = new float[64];

        var packed = quantizer.Quantize(zero);
        Assert.Equal(0f, packed.Norm);

        var restored = quantizer.Dequantize(packed);
        Assert.All(restored, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void NearZeroVector_HandledGracefully()
    {
        var quantizer = new TurboQuantMSE(64, 4, seed: 42);
        var tiny = new float[64];
        tiny[0] = 1e-15f;

        var packed = quantizer.Quantize(tiny);
        Assert.Equal(0f, packed.Norm); // below epsilon, treated as zero

        var restored = quantizer.Dequantize(packed);
        Assert.All(restored, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void SparseVector_Works()
    {
        var quantizer = new TurboQuantMSE(256, 4, seed: 42);
        var sparse = new float[256];
        sparse[0] = 1f; // only one non-zero element

        var packed = quantizer.Quantize(sparse);
        var restored = quantizer.Dequantize(packed);

        var cosine = DotProductSimd.CosineSimilarity(sparse, restored);
        _out.WriteLine($"Sparse vector cosine: {cosine:F6}");
        Assert.True(cosine > 0.90);
    }

    [Fact]
    public void ConstantVector_Works()
    {
        var quantizer = new TurboQuantMSE(64, 4, seed: 42);
        var constant = Enumerable.Repeat(1f / MathF.Sqrt(64), 64).ToArray();

        var packed = quantizer.Quantize(constant);
        var restored = quantizer.Dequantize(packed);

        var cosine = DotProductSimd.CosineSimilarity(constant, restored);
        _out.WriteLine($"Constant vector cosine: {cosine:F6}");
        Assert.True(cosine > 0.95);
    }

    // ──────────────────────────────────────────────
    // 5. Codebook Edge Cases
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 2)]
    [InlineData(4, 16)]
    [InlineData(4, 4096)]
    public void Codebook_VariousDimensions_DoesNotCrash(int bits, int dim)
    {
        var cb = new BetaCodebook(bits, dim);
        Assert.Equal(1 << bits, cb.Levels);
        Assert.Equal(dim, cb.Dimension);

        // Centroids should be finite and ordered
        var centroids = cb.Centroids.ToArray();
        for (var i = 0; i < centroids.Length; i++)
        {
            Assert.True(float.IsFinite(centroids[i]));
            if (i > 0) Assert.True(centroids[i] > centroids[i - 1], $"Centroids not sorted at index {i}");
        }
    }

    [Fact]
    public void Codebook_InvalidBits_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BetaCodebook(0, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BetaCodebook(1, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BetaCodebook(5, 64));
    }

    [Fact]
    public void Codebook_InvalidDimension_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BetaCodebook(4, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BetaCodebook(4, 0));
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    private static float[] RandomUnitVector(int dim, Random rng)
    {
        var v = new float[dim];
        var norm = 0f;
        for (var i = 0; i < dim; i++)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = rng.NextDouble();
            v[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            norm += v[i] * v[i];
        }
        norm = MathF.Sqrt(norm);
        for (var i = 0; i < dim; i++) v[i] /= norm;
        return v;
    }
}
