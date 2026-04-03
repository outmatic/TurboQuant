namespace TurboQuant.Diagnostics;

/// <summary>
/// Validation tool that verifies quantization MSE against theoretical values from the TurboQuant paper.
/// </summary>
public static class QuantizationBenchmark
{
    /// <summary>
    /// Theoretical D_mse = d · C(f_X, b) values from the paper.
    /// For large d, D_mse ≈ MSE_{N(0,1)}(b) (the Lloyd-Max distortion for standard normal).
    /// These are approximately dimension-independent.
    /// </summary>
    private static readonly Dictionary<int, double> TheoreticalDmse = new()
    {
        [2] = 0.11753,  // 2-bit
        [3] = 0.03454,  // 3-bit
        [4] = 0.009497, // 4-bit
    };

    /// <summary>
    /// Runs a validation benchmark and compares empirical D_mse against theoretical values.
    /// </summary>
    /// <param name="dim">Vector dimension to test.</param>
    /// <param name="bits">Bit width to test.</param>
    /// <param name="numVectors">Number of random vectors to use (default 10000).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public static ValidationResult RunValidation(int dim = 768, int bits = 4, int numVectors = 10000, int seed = 42)
    {
        var quantizer = new Core.Quantizers.TurboQuantMSE(dim, bits, seed);
        var rng = new Random(seed + 1);

        var totalDmse = 0.0;

        for (int v = 0; v < numVectors; v++)
        {
            var vector = new float[dim];
            var norm = 0f;
            for (int i = 0; i < dim; i++)
            {
                vector[i] = (float)NextGaussian(rng);
                norm += vector[i] * vector[i];
            }
            norm = MathF.Sqrt(norm);
            for (int i = 0; i < dim; i++)
                vector[i] /= norm;

            var packed = quantizer.Quantize(vector);
            var restored = quantizer.Dequantize(packed);

            // D_mse = ||x - x̃||² (total squared error on unit vector)
            var dmse = 0.0;
            for (int i = 0; i < dim; i++)
            {
                double diff = vector[i] - restored[i];
                dmse += diff * diff;
            }
            totalDmse += dmse;
        }

        var empiricalDmse = totalDmse / numVectors;
        var theoreticalDmse = TheoreticalDmse.GetValueOrDefault(bits, 0);
        var deltaPercent = theoreticalDmse > 0
            ? Math.Abs(empiricalDmse - theoreticalDmse) / theoreticalDmse * 100.0
            : double.NaN;

        return new ValidationResult(
            bits, dim, numVectors, empiricalDmse, theoreticalDmse, deltaPercent,
            Passed: empiricalDmse <= theoreticalDmse * 1.1); // must not be worse than theoretical + 10%
    }

    private static double NextGaussian(Random rng) => Core.MathUtils.NextGaussian(rng);
}

/// <summary>
/// Result of a validation benchmark run.
/// </summary>
public readonly record struct ValidationResult(
    int Bits,
    int Dimension,
    int NumVectors,
    double EmpiricalMSE,
    double TheoreticalMSE,
    double DeltaPercent,
    bool Passed)
{
    /// <inheritdoc/>
    public override string ToString() =>
        $"TurboQuant Validation [{Bits}-bit, dim={Dimension}, n={NumVectors}]: " +
        $"D_mse={EmpiricalMSE:F6}, Theoretical={TheoreticalMSE:F6}, " +
        $"Delta={DeltaPercent:F2}%. {(Passed ? "PASS" : "FAIL")}";
}
