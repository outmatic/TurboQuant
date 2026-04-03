using System.Runtime.CompilerServices;

namespace TurboQuant.Core.Codebook;

/// <summary>
/// Lloyd-Max optimal scalar quantizer for the marginal distribution of a uniform
/// point on the unit hypersphere S^{d-1}. Centroids are computed via the iterative
/// Lloyd-Max algorithm applied to f_X(x) = C·(1-x²)^((d-3)/2) on [-1,1]
/// (the Beta(d/2, 1/2) distribution), as specified in the TurboQuant paper
/// (Zandieh et al., ICLR 2026).
/// </summary>
public sealed class BetaCodebook : ICodebook
{
    // Standard N(0,1) Lloyd-Max centroids used as initialization (positive half only).
    // The full set is symmetric: {-c_K, ..., -c_1, c_1, ..., c_K}.
    private static readonly double[] GaussInit2 = { 0.452780, 1.510418 };
    private static readonly double[] GaussInit3 = { 0.245114, 0.756009, 1.343970, 2.151968 };
    private static readonly double[] GaussInit4 = { 0.128437, 0.388115, 0.656806, 0.942390,
                                                     1.256230, 1.618026, 2.069014, 2.732645 };

    private readonly float[] _centroids;
    private readonly float[] _boundaries;
    private readonly float[] _dotLut;    // [i * levels + j] = c_i * c_j
    private readonly float[] _normSqLut; // [i] = c_i²
    private readonly int _bits;
    private readonly int _dimension;

    /// <inheritdoc/>
    public int Bits => _bits;
    /// <inheritdoc/>
    public int Levels => _centroids.Length;
    /// <inheritdoc/>
    public ReadOnlySpan<float> Centroids => _centroids;
    /// <summary>Dimension for which this codebook was optimized.</summary>
    public int Dimension => _dimension;

    /// <summary>
    /// Constructs a codebook by running Lloyd-Max on the exact Beta(d/2, 1/2) PDF.
    /// </summary>
    /// <param name="bits">Bit width (2, 3, or 4).</param>
    /// <param name="dimension">Vector dimension d (determines the Beta distribution shape).</param>
    public BetaCodebook(int bits, int dimension)
    {
        _bits = Guard.Bits(bits);
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 2);
        _dimension = dimension;
        var levels = 1 << bits;
        var halfExp = (dimension - 3.0) / 2.0;

        // Initialize centroids from Gaussian approximation scaled by sigma = 1/sqrt(d).
        // For large d, Beta(d/2, 1/2) ~ N(0, 1/d), so this is an excellent starting point.
        var sigma = 1.0 / Math.Sqrt(dimension);
        var gaussHalf = bits switch
        {
            2 => GaussInit2,
            3 => GaussInit3,
            4 => GaussInit4,
            _ => throw new ArgumentOutOfRangeException(nameof(bits))
        };

        var halfLevels = levels / 2;
        var centroids = new double[levels];
        for (var i = 0; i < halfLevels; i++)
        {
            centroids[halfLevels + i] = gaussHalf[i] * sigma;       // positive half
            centroids[halfLevels - 1 - i] = -gaussHalf[i] * sigma;  // negative half (mirror)
        }

        // Lloyd-Max iterations with exact Beta PDF
        var boundaries = new double[levels - 1];

        for (var iter = 0; iter < 100; iter++)
        {
            // Step 1: Update boundaries (midpoints between adjacent centroids)
            for (var i = 0; i < levels - 1; i++)
                boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;

            // Step 2: Update centroids = E[X | b_{i-1} < X < b_i] under f_X
            var newCentroids = new double[levels];
            var maxDiff = 0.0;

            for (var i = 0; i < levels; i++)
            {
                var lo = i == 0 ? -1.0 : boundaries[i - 1];
                var hi = i == levels - 1 ? 1.0 : boundaries[i];

                var moment = AnalyticFirstMoment(lo, hi, halfExp);
                var mass = NumericalMass(lo, hi, halfExp);

                // If mass is negligible (tail bin), use midpoint as fallback
                newCentroids[i] = mass > 1e-300 ? moment / mass : (lo + hi) / 2.0;
                maxDiff = Math.Max(maxDiff, Math.Abs(newCentroids[i] - centroids[i]));
            }

            Array.Copy(newCentroids, centroids, levels);
            if (maxDiff < 1e-14) break;
        }

        // Final boundary update
        for (var i = 0; i < levels - 1; i++)
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;

        // Convert to float32
        _centroids = new float[levels];
        _boundaries = new float[levels - 1];
        for (var i = 0; i < levels; i++) _centroids[i] = (float)centroids[i];
        for (var i = 0; i < levels - 1; i++) _boundaries[i] = (float)boundaries[i];

        // Precompute dot product and norm² lookup tables
        _dotLut = new float[levels * levels];
        _normSqLut = new float[levels];
        for (var i = 0; i < levels; i++)
        {
            _normSqLut[i] = _centroids[i] * _centroids[i];
            for (var j = 0; j < levels; j++)
                _dotLut[i * levels + j] = _centroids[i] * _centroids[j];
        }
    }

    /// <summary>Dot product between centroids i and j, via precomputed LUT.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal float DotLookup(int i, int j) => _dotLut[i * Levels + j];

    /// <summary>Squared norm of centroid i, via precomputed LUT.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal float NormSqLookup(int i) => _normSqLut[i];

    /// <summary>
    /// Analytic formula for ∫_a^b x·(1-x²)^n dx where n = halfExp.
    /// Derivation: let u = 1-x², du = -2x dx → ∫ x·u^n dx = -u^(n+1)/(2(n+1)).
    /// Result: [(1-a²)^(n+1) - (1-b²)^(n+1)] / (2(n+1)).
    /// </summary>
    private static double AnalyticFirstMoment(double a, double b, double halfExp)
    {
        var exp1 = halfExp + 1.0;
        var tA = 1.0 - a * a;
        var tB = 1.0 - b * b;
        var termA = tA > 0 ? Math.Pow(tA, exp1) : 0.0;
        var termB = tB > 0 ? Math.Pow(tB, exp1) : 0.0;
        return (termA - termB) / (2.0 * exp1);
    }

    /// <summary>
    /// Numerical integration of ∫_a^b (1-x²)^halfExp dx using composite Simpson's rule
    /// with 256 subintervals. The unnormalized Beta PDF (1-x²)^((d-3)/2) is smooth and
    /// well-behaved on [-1,1]; values that underflow to 0 for large exponents are handled
    /// naturally since Math.Pow returns 0.
    /// </summary>
    private static double NumericalMass(double a, double b, double halfExp)
    {
        const int n = 256; // must be even
        var h = (b - a) / n;
        if (h <= 0) return 0;

        var sum = BetaPdfUnnorm(a, halfExp) + BetaPdfUnnorm(b, halfExp);
        for (var i = 1; i < n; i += 2)
            sum += 4.0 * BetaPdfUnnorm(a + i * h, halfExp);
        for (var i = 2; i < n; i += 2)
            sum += 2.0 * BetaPdfUnnorm(a + i * h, halfExp);

        return sum * h / 3.0;
    }

    /// <summary>
    /// Unnormalized Beta PDF: (1-x²)^halfExp. The normalization constant cancels
    /// in the centroid computation (E[X|a&lt;X&lt;b] = moment/mass).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double BetaPdfUnnorm(double x, double halfExp)
    {
        var t = 1.0 - x * x;
        return t > 0 ? Math.Pow(t, halfExp) : 0.0;
    }

    /// <summary>
    /// Finds nearest centroid using binary search on sorted boundaries. O(log levels).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Quantize(float value)
    {
        var lo = 0;
        var hi = _boundaries.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) >> 1;
            if (value > _boundaries[mid])
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Dequantize(int index) => _centroids[index];
}
