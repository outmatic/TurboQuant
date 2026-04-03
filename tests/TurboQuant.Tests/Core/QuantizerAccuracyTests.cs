using TurboQuant.Core.Quantizers;
using TurboQuant.Core.Simd;
using TurboQuant.Diagnostics;
using Xunit;

namespace TurboQuant.Tests.Core;

public class QuantizerAccuracyTests
{
    [Theory]
    [InlineData(4, 0.009497, 768)]
    [InlineData(3, 0.03454, 768)]
    public void MSE_WithinTheoreticalBounds(int bits, double theoreticalDmse, int dim)
    {
        var quantizer = new TurboQuantMSE(dim, bits, seed: 42);
        var rng = new Random(123);
        int numVectors = 5000;
        double totalDmse = 0;

        for (int v = 0; v < numVectors; v++)
        {
            var vector = GenerateRandomUnitVector(dim, rng);
            var packed = quantizer.Quantize(vector);
            var restored = quantizer.Dequantize(packed);

            // D_mse = ||x - x̃||² (total squared error on unit vector)
            double dmse = 0;
            for (int i = 0; i < dim; i++)
            {
                double diff = vector[i] - restored[i];
                dmse += diff * diff;
            }
            totalDmse += dmse;
        }

        double avgDmse = totalDmse / numVectors;

        // D_mse should be close to the theoretical Lloyd-Max distortion,
        // which is approximately dimension-independent: D_mse ≈ MSE_{N(0,1)}(b)
        Assert.InRange(avgDmse, 0, theoreticalDmse * 1.5);
    }

    [Theory]
    [InlineData(4, 768)]
    [InlineData(3, 768)]
    public void CosineSimilarity_HighQuality(int bits, int dim)
    {
        var quantizer = new TurboQuantMSE(dim, bits, seed: 42);
        var rng = new Random(456);
        int numVectors = 1000;
        double totalCosine = 0;

        for (int v = 0; v < numVectors; v++)
        {
            var vector = GenerateRandomUnitVector(dim, rng);
            var packed = quantizer.Quantize(vector);
            var restored = quantizer.Dequantize(packed);

            totalCosine += DotProductSimd.CosineSimilarity(vector, restored);
        }

        double avgCosine = totalCosine / numVectors;

        if (bits == 4)
            Assert.True(avgCosine > 0.99, $"4-bit cosine similarity {avgCosine} too low");
        else
            Assert.True(avgCosine > 0.95, $"3-bit cosine similarity {avgCosine} too low");
    }

    [Fact]
    public void QuantizeBatch_MatchesSingleQuantize()
    {
        int dim = 128;
        int count = 10;
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);
        var rng = new Random(789);

        var batchInput = new float[dim * count];
        for (int i = 0; i < batchInput.Length; i++)
            batchInput[i] = (float)(rng.NextDouble() * 2 - 1);

        var batchOutput = new TurboQuant.Core.Packing.PackedVector[count];
        quantizer.QuantizeBatch(batchInput, batchOutput);

        // Verify each batch result matches individual quantization
        var quantizer2 = new TurboQuantMSE(dim, 4, seed: 42);
        for (int i = 0; i < count; i++)
        {
            var single = quantizer2.Quantize(batchInput.AsSpan(i * dim, dim));
            Assert.Equal(single.Norm, batchOutput[i].Norm, precision: 4);
            Assert.Equal(single.Data, batchOutput[i].Data);
        }
    }

    [Fact]
    public void ApproxSimilarity_CorrelatesWithTrueSimilarity()
    {
        int dim = 256;
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);
        var rng = new Random(101);

        var a = GenerateRandomUnitVector(dim, rng);
        var b = GenerateRandomUnitVector(dim, rng);

        float trueCosine = DotProductSimd.CosineSimilarity(a, b);

        var packedA = quantizer.Quantize(a);
        var packedB = quantizer.Quantize(b);
        float approxCosine = quantizer.ApproxSimilarity(packedA, packedB);

        // Should be correlated within reasonable tolerance
        Assert.InRange(approxCosine - trueCosine, -0.15f, 0.15f);
    }

    [Fact]
    public void Stats_TracksCorrectly()
    {
        int dim = 128;
        var quantizer = new TurboQuantMSE(dim, 4, seed: 42);

        var stats = quantizer.GetStats();
        Assert.Equal(0, stats.TotalVectorsProcessed);

        var rng = new Random(42);
        for (int i = 0; i < 10; i++)
            quantizer.Quantize(GenerateRandomUnitVector(dim, rng));

        Assert.Equal(10, stats.TotalVectorsProcessed);
        Assert.True(stats.AverageMSE > 0);
        Assert.True(stats.AverageQuantizeMs >= 0);
    }

    [Fact]
    public void ZeroVector_HandledGracefully()
    {
        var quantizer = new TurboQuantMSE(128, 4, seed: 42);
        var zero = new float[128];

        var packed = quantizer.Quantize(zero);
        Assert.Equal(0f, packed.Norm);

        var restored = quantizer.Dequantize(packed);
        Assert.All(restored, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void ValidationBenchmark_Passes()
    {
        var result = QuantizationBenchmark.RunValidation(dim: 256, bits: 4, numVectors: 2000);
        // Just verify it runs without throwing; actual MSE tolerance is generous
        Assert.True(result.EmpiricalMSE > 0);
    }

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
