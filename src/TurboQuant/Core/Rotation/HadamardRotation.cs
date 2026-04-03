using System.Numerics;
using System.Runtime.CompilerServices;

namespace TurboQuant.Core.Rotation;

/// <summary>
/// Walsh-Hadamard Transform with deterministic sign flips for randomized rotation.
/// O(d log d) complexity. Pads input to next power of 2 if needed.
/// </summary>
public sealed class HadamardRotation : IRotation
{
    private readonly int _originalDim;
    private readonly int _paddedDim;
    private readonly float[] _signs;  // +1 or -1, length = _paddedDim
    private readonly float _scale;

    /// <inheritdoc/>
    public int PaddedDimension => _paddedDim;

    /// <param name="dimension">Original vector dimension.</param>
    /// <param name="seed">Seed for deterministic sign flips.</param>
    public HadamardRotation(int dimension, int seed = 42)
    {
        _originalDim = Guard.Dimension(dimension);
        _paddedDim = NextPowerOfTwo(dimension);
        _scale = 1f / MathF.Sqrt(_paddedDim);
        _signs = new float[_paddedDim];

        var rng = new Random(seed);
        for (var i = 0; i < _paddedDim; i++)
            _signs[i] = rng.Next(2) == 0 ? 1f : -1f;
    }

    /// <summary>
    /// Applies sign flips then in-place WHT. Input span must be at least PaddedDimension long.
    /// If shorter, caller should pad with zeros.
    /// </summary>
    [SkipLocalsInit]
    public void Transform(Span<float> vector)
    {
        Guard.VectorLength(vector.Length, _paddedDim);

        // Apply sign flips
        for (var i = 0; i < _paddedDim; i++)
            vector[i] *= _signs[i];

        // WHT butterfly
        WhtInPlace(vector[.._paddedDim]);

        // Scale
        for (var i = 0; i < _paddedDim; i++)
            vector[i] *= _scale;
    }

    /// <summary>
    /// Inverse transform: WHT is self-inverse up to scaling. Scale then WHT then un-flip signs.
    /// </summary>
    [SkipLocalsInit]
    public void InverseTransform(Span<float> vector)
    {
        Guard.VectorLength(vector.Length, _paddedDim);

        // WHT matrix H satisfies H * H = n * I, so H_normalized = H/sqrt(n), and H_norm * H_norm = I.
        // So inverse is just: apply WHT again with same scale.

        // WHT butterfly
        WhtInPlace(vector[.._paddedDim]);

        // Scale
        for (var i = 0; i < _paddedDim; i++)
            vector[i] *= _scale;

        // Undo sign flips (signs are +/-1, so self-inverse)
        for (var i = 0; i < _paddedDim; i++)
            vector[i] *= _signs[i];
    }

    /// <summary>
    /// Standard iterative WHT butterfly, O(n log n).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WhtInPlace(Span<float> data)
    {
        var n = data.Length;
        for (var halfSize = 1; halfSize < n; halfSize <<= 1)
        {
            for (var i = 0; i < n; i += halfSize << 1)
            {
                for (var j = i; j < i + halfSize; j++)
                {
                    var a = data[j];
                    var b = data[j + halfSize];
                    data[j] = a + b;
                    data[j + halfSize] = a - b;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int NextPowerOfTwo(int v) =>
        (int)BitOperations.RoundUpToPowerOf2((uint)v);
}
