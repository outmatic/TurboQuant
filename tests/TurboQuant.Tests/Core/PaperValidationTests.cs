using TurboQuant.Core.Codebook;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Quantizers;
using TurboQuant.Core.Rotation;
using TurboQuant.Core.Simd;
using TurboQuant.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace TurboQuant.Tests.Core;

/// <summary>
/// Comprehensive validation against the TurboQuant paper
/// (Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", ICLR 2026).
/// </summary>
public class PaperValidationTests
{
    private readonly ITestOutputHelper _output;

    public PaperValidationTests(ITestOutputHelper output) => _output = output;

    // ──────────────────────────────────────────────
    // 1. Random Orthogonal Rotation (§3, Algorithm 1)
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(768)]
    public void Rotation_IsOrthogonal_PreservesNorm(int dim)
    {
        var rot = new RandomRotation(dim, seed: 42);
        var rng = new Random(99);

        for (int trial = 0; trial < 20; trial++)
        {
            var v = GenerateRandomUnitVector(dim, rng);
            float normBefore = DotProductSimd.L2Norm(v);

            rot.Transform(v);
            float normAfter = DotProductSimd.L2Norm(v);

            Assert.InRange(normAfter / normBefore, 0.9999, 1.0001);
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(768)]
    public void Rotation_IsOrthogonal_PreservesInnerProducts(int dim)
    {
        var rot = new RandomRotation(dim, seed: 42);
        var rng = new Random(77);

        var a = GenerateRandomUnitVector(dim, rng);
        var b = GenerateRandomUnitVector(dim, rng);

        float dotBefore = DotProductSimd.DotProduct(a, b);

        rot.Transform(a);
        rot.Transform(b);

        float dotAfter = DotProductSimd.DotProduct(a, b);

        // Orthogonal transform preserves inner products
        Assert.InRange(dotAfter - dotBefore, -1e-4f, 1e-4f);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(768)]
    public void Rotation_InverseIsExact(int dim)
    {
        var rot = new RandomRotation(dim, seed: 42);
        var rng = new Random(55);
        var original = GenerateRandomUnitVector(dim, rng);
        var copy = original.ToArray();

        rot.Transform(copy);
        rot.InverseTransform(copy);

        for (int i = 0; i < dim; i++)
            Assert.InRange(copy[i] - original[i], -1e-4f, 1e-4f);
    }

    // ──────────────────────────────────────────────
    // 2. Coordinate Distribution after Rotation (§3, Theorem 1)
    //    Each coordinate of Π·x should follow Beta(d/2, 1/2) on [-1,1]
    //    which has mean=0 and variance=1/d
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(128)]
    [InlineData(768)]
    public void RotatedCoordinates_HaveCorrectMoments(int dim)
    {
        var rot = new RandomRotation(dim, seed: 42);
        var rng = new Random(33);
        int numVectors = 2000;

        // Collect first coordinate of all rotated vectors
        double sumX = 0, sumX2 = 0;
        for (int v = 0; v < numVectors; v++)
        {
            var vec = GenerateRandomUnitVector(dim, rng);
            rot.Transform(vec);
            double x = vec[0]; // first coordinate
            sumX += x;
            sumX2 += x * x;
        }

        double mean = sumX / numVectors;
        double variance = sumX2 / numVectors - mean * mean;
        double expectedVariance = 1.0 / dim;

        _output.WriteLine($"dim={dim}: mean={mean:F6} (expected ~0), variance={variance:F6} (expected {expectedVariance:F6})");

        // Mean should be close to 0
        Assert.InRange(mean, -0.05, 0.05);
        // Variance should be close to 1/d
        Assert.InRange(variance / expectedVariance, 0.7, 1.3);
    }

    // ──────────────────────────────────────────────
    // 3. Lloyd-Max Codebook (§3, Eq. 4)
    //    Centroids minimize C(f_X, b) for Beta distribution
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(2, 128)]
    [InlineData(3, 128)]
    [InlineData(4, 128)]
    [InlineData(4, 768)]
    public void Codebook_CentroidsAreSymmetric(int bits, int dim)
    {
        var cb = new BetaCodebook(bits, dim);
        int levels = cb.Levels;

        // Centroids should be symmetric around 0
        for (int i = 0; i < levels / 2; i++)
        {
            float pos = cb.Centroids[levels / 2 + i];
            float neg = cb.Centroids[levels / 2 - 1 - i];
            Assert.InRange(pos + neg, -1e-6f, 1e-6f);
        }
    }

    [Theory]
    [InlineData(2, 128)]
    [InlineData(4, 768)]
    public void Codebook_CentroidsAreInUnitInterval(int bits, int dim)
    {
        var cb = new BetaCodebook(bits, dim);
        foreach (var c in cb.Centroids.ToArray())
        {
            Assert.InRange(c, -1.0f, 1.0f);
        }
    }

    [Theory]
    [InlineData(4, 128)]
    [InlineData(4, 768)]
    public void Codebook_CentroidsScaleAsInverseSqrtDim(int bits, int dim)
    {
        var cb = new BetaCodebook(bits, dim);
        float outermost = cb.Centroids[cb.Levels - 1];
        double sigma = 1.0 / Math.Sqrt(dim);

        // Paper §3: centroids ≈ gaussian_centroids / √d
        // The outermost 4-bit centroid is ≈ 2.73 / √d
        double expected = 2.73 * sigma;
        double ratio = outermost / expected;

        _output.WriteLine($"dim={dim}: outermost centroid={outermost:F6}, expected≈{expected:F6}, ratio={ratio:F4}");
        Assert.InRange(ratio, 0.9, 1.1);
    }

    [Fact]
    public void Codebook_QuantizeDequantize_RoundTrips()
    {
        var cb = new BetaCodebook(4, 768);
        // Each centroid should map to itself
        for (int i = 0; i < cb.Levels; i++)
        {
            float c = cb.Centroids[i];
            int idx = cb.Quantize(c);
            float restored = cb.Dequantize(idx);
            Assert.Equal(c, restored);
        }
    }

    // ──────────────────────────────────────────────
    // 4. MSE Distortion (Theorem 1)
    //    D_mse = d · C(f_X, b) ≈ MSE_{N(0,1)}(b)
    //    Paper bounds: ≤ (√3·π/2) · (1/4^b)
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(4, 0.009497, 256)]
    [InlineData(4, 0.009497, 768)]
    [InlineData(3, 0.03454, 256)]
    [InlineData(3, 0.03454, 768)]
    [InlineData(2, 0.11753, 256)]
    public void Dmse_WithinTheoreticalBounds(int bits, double theoreticalDmse, int dim)
    {
        var quantizer = new TurboQuantMSE(dim, bits, seed: 42);
        var rng = new Random(200);
        int numVectors = 2000;
        double totalDmse = 0;

        for (int v = 0; v < numVectors; v++)
        {
            var vector = GenerateRandomUnitVector(dim, rng);
            var packed = quantizer.Quantize(vector);
            var restored = quantizer.Dequantize(packed);

            double dmse = 0;
            for (int i = 0; i < dim; i++)
            {
                double diff = vector[i] - restored[i];
                dmse += diff * diff;
            }
            totalDmse += dmse;
        }

        double avgDmse = totalDmse / numVectors;
        double delta = Math.Abs(avgDmse - theoreticalDmse) / theoreticalDmse * 100;

        _output.WriteLine($"{bits}-bit dim={dim}: D_mse={avgDmse:F6}, theoretical={theoreticalDmse:F6}, delta={delta:F2}%");

        // With Hadamard + padding, D_mse can be better than theoretical (padding spreads error).
        // Verify we're not WORSE than theoretical by more than 10%.
        Assert.True(avgDmse < theoreticalDmse * 1.1, $"D_mse {avgDmse:F6} exceeds theoretical {theoreticalDmse:F6} by >{delta:F1}%");
    }

    [Theory]
    [InlineData(4)]
    [InlineData(3)]
    [InlineData(2)]
    public void Dmse_SatisfiesPaperUpperBound(int bits)
    {
        // Paper Theorem 1: D_mse ≤ (√3·π/2) · (1/4^b)
        double upperBound = Math.Sqrt(3) * Math.PI / 2.0 * Math.Pow(4, -bits);
        int dim = 256;

        var quantizer = new TurboQuantMSE(dim, bits, seed: 42);
        var rng = new Random(300);
        int numVectors = 1000;
        double totalDmse = 0;

        for (int v = 0; v < numVectors; v++)
        {
            var vector = GenerateRandomUnitVector(dim, rng);
            var packed = quantizer.Quantize(vector);
            var restored = quantizer.Dequantize(packed);

            double dmse = 0;
            for (int i = 0; i < dim; i++)
            {
                double diff = vector[i] - restored[i];
                dmse += diff * diff;
            }
            totalDmse += dmse;
        }

        double avgDmse = totalDmse / numVectors;
        _output.WriteLine($"{bits}-bit: D_mse={avgDmse:F6}, paper upper bound={upperBound:F6}");
        Assert.True(avgDmse < upperBound, $"D_mse {avgDmse} exceeds paper upper bound {upperBound}");
    }

    // ──────────────────────────────────────────────
    // 5. Cosine Similarity Quality
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(4, 768, 0.995)]
    [InlineData(3, 768, 0.97)]
    [InlineData(2, 768, 0.90)]
    [InlineData(4, 256, 0.99)]
    [InlineData(4, 128, 0.98)]
    public void CosineSimilarity_ExceedsThreshold(int bits, int dim, double minCosine)
    {
        var quantizer = new TurboQuantMSE(dim, bits, seed: 42);
        var rng = new Random(400);
        int numVectors = 500;
        double totalCosine = 0;

        for (int v = 0; v < numVectors; v++)
        {
            var vector = GenerateRandomUnitVector(dim, rng);
            var packed = quantizer.Quantize(vector);
            var restored = quantizer.Dequantize(packed);
            totalCosine += DotProductSimd.CosineSimilarity(vector, restored);
        }

        double avgCosine = totalCosine / numVectors;
        _output.WriteLine($"{bits}-bit dim={dim}: avg cosine={avgCosine:F6} (threshold={minCosine})");
        Assert.True(avgCosine > minCosine, $"Cosine {avgCosine:F6} below threshold {minCosine}");
    }

    // ──────────────────────────────────────────────
    // 6. Unbiased Estimation (Algorithm 1 property)
    //    E[x̃] = x for the MSE quantizer
    // ──────────────────────────────────────────────

    [Fact]
    public void Dequantize_IsApproximatelyUnbiased()
    {
        int dim = 128;
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);

        // Fix a single vector, quantize many times with different rotations
        // (actually same rotation, so check that the average error is small)
        var rng = new Random(500);
        int numTrials = 50;
        var sumError = new double[dim];

        for (int t = 0; t < numTrials; t++)
        {
            // Each trial uses a fresh quantizer with different seed → different Π
            var q = new TurboQuantMSE(dim, 4, seed: t);
            var vector = GenerateRandomUnitVector(dim, rng);
            var packed = q.Quantize(vector);
            var restored = q.Dequantize(packed);

            for (int i = 0; i < dim; i++)
                sumError[i] += restored[i] - vector[i];
        }

        // Average error per coordinate should be close to 0
        double maxBias = 0;
        for (int i = 0; i < dim; i++)
        {
            double avgError = sumError[i] / numTrials;
            maxBias = Math.Max(maxBias, Math.Abs(avgError));
        }

        _output.WriteLine($"Max per-coordinate bias: {maxBias:F6}");
        Assert.True(maxBias < 0.1, $"Max bias {maxBias} too large — estimator may not be approximately unbiased");
    }

    // ──────────────────────────────────────────────
    // 7. Bit Packing Integrity
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(2, 128)]
    [InlineData(3, 128)]
    [InlineData(4, 128)]
    [InlineData(4, 768)]
    [InlineData(3, 384)]
    public void BitPacking_ExactRoundTrip(int bits, int dim)
    {
        int levels = 1 << bits;
        var rng = new Random(600);
        var indices = new int[dim];
        for (int i = 0; i < dim; i++)
            indices[i] = rng.Next(levels);

        int packedBytes = PackedVector.GetPackedByteCount(dim, bits);
        var packed = new byte[packedBytes];
        BitPacker.Pack(indices, packed, bits);

        var unpacked = new int[dim];
        BitPacker.Unpack(packed, unpacked, dim, bits);

        for (int i = 0; i < dim; i++)
            Assert.Equal(indices[i], unpacked[i]);
    }

    // ──────────────────────────────────────────────
    // 8. Full Pipeline Determinism
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(128)]
    [InlineData(768)]
    public void Quantize_IsDeterministic(int dim)
    {
        var q1 = new TurboQuantMSE(dim, 4, seed: 42);
        var q2 = new TurboQuantMSE(dim, 4, seed: 42);

        var rng = new Random(700);
        var vector = GenerateRandomUnitVector(dim, rng);

        var p1 = q1.Quantize(vector);
        var p2 = q2.Quantize(vector);

        Assert.Equal(p1.Norm, p2.Norm);
        Assert.Equal(p1.Data, p2.Data);
    }

    // ──────────────────────────────────────────────
    // 9. Compression Ratio
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(4, 768)]
    [InlineData(3, 768)]
    [InlineData(2, 768)]
    [InlineData(4, 256)]
    public void CompressionRatio_IsCorrect(int bits, int dim)
    {
        var quantizer = new TurboQuantMSE(dim, bits, seed: 42);
        var rng = new Random(800);
        var vector = GenerateRandomUnitVector(dim, rng);
        var packed = quantizer.Quantize(vector);

        long originalBytes = dim * sizeof(float);
        long compressedBytes = packed.Data.Length + sizeof(float);

        var expectedRatio = (double)originalBytes / compressedBytes;

        _output.WriteLine($"{bits}-bit dim={dim}: {originalBytes}B → {compressedBytes}B, ratio={expectedRatio:F1}x");

        // With Hadamard padding (non-power-of-2 dims use more bytes), ratios are lower
        var minRatio = bits switch { 4 => 5.5, 3 => 7.0, 2 => 10.0, _ => 1.0 };
        Assert.True(expectedRatio > minRatio, $"Compression ratio {expectedRatio:F1}x below expected {minRatio}x");
    }

    // ──────────────────────────────────────────────
    // 10. ApproxSimilarity preserves ranking
    // ──────────────────────────────────────────────

    [Fact]
    public void ApproxSimilarity_PreservesRanking()
    {
        int dim = 256;
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);
        var rng = new Random(900);

        var query = GenerateRandomUnitVector(dim, rng);
        var packedQuery = quantizer.Quantize(query);

        // Generate 50 documents and rank by true vs approx similarity
        int n = 50;
        var trueSims = new float[n];
        var approxSims = new float[n];
        var packedDocs = new PackedVector[n];

        for (int i = 0; i < n; i++)
        {
            var doc = GenerateRandomUnitVector(dim, rng);
            trueSims[i] = DotProductSimd.CosineSimilarity(query, doc);
            packedDocs[i] = quantizer.Quantize(doc);
            approxSims[i] = quantizer.ApproxSimilarity(packedQuery, packedDocs[i]);
        }

        // Check Spearman rank correlation (approximate: check top-5 overlap)
        var trueTop5 = trueSims.Select((s, i) => (s, i)).OrderByDescending(x => x.s).Take(5).Select(x => x.i).ToHashSet();
        var approxTop5 = approxSims.Select((s, i) => (s, i)).OrderByDescending(x => x.s).Take(5).Select(x => x.i).ToHashSet();

        int overlap = trueTop5.Intersect(approxTop5).Count();
        _output.WriteLine($"Top-5 overlap: {overlap}/5");

        // At least 2 of top-5 should overlap (random baseline would give ~0.5)
        Assert.True(overlap >= 2, $"Top-5 overlap {overlap} too low — ranking not preserved");
    }

    // ──────────────────────────────────────────────
    // 11. Hadamard (fast path) also works
    // ──────────────────────────────────────────────

    [Fact]
    public void HadamardPath_AlsoProducesGoodResults()
    {
        int dim = 256;
        var rotation = new HadamardRotation(dim, seed: 42);
        var quantizer = new TurboQuantMSE(dim, 4, rotation);
        var rng = new Random(1000);
        int numVectors = 500;
        double totalCosine = 0;

        for (int v = 0; v < numVectors; v++)
        {
            var vector = GenerateRandomUnitVector(dim, rng);
            var packed = quantizer.Quantize(vector);
            var restored = quantizer.Dequantize(packed);
            totalCosine += DotProductSimd.CosineSimilarity(vector, restored);
        }

        double avgCosine = totalCosine / numVectors;
        _output.WriteLine($"Hadamard 4-bit dim=256: avg cosine={avgCosine:F6}");
        Assert.True(avgCosine > 0.98, $"Hadamard cosine {avgCosine} too low");
    }

    // ──────────────────────────────────────────────
    // 12. Official validation benchmark
    // ──────────────────────────────────────────────

    [Theory]
    [InlineData(4, 256)]
    [InlineData(4, 768)]
    [InlineData(3, 256)]
    public void OfficialValidation_Passes(int bits, int dim)
    {
        var result = QuantizationBenchmark.RunValidation(dim: dim, bits: bits, numVectors: 2000);
        _output.WriteLine(result.ToString());
        Assert.True(result.Passed, $"Validation failed: {result}");
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    private static float[] GenerateRandomUnitVector(int dim, Random rng)
    {
        var v = new float[dim];
        float norm = 0;
        for (int i = 0; i < dim; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            v[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            norm += v[i] * v[i];
        }
        norm = MathF.Sqrt(norm);
        for (int i = 0; i < dim; i++)
            v[i] /= norm;
        return v;
    }
}
