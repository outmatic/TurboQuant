using System.Numerics;
using System.Runtime.CompilerServices;

namespace TurboQuant.Core.Rotation;

/// <summary>
/// Walsh-Hadamard Transform with deterministic sign flips for randomized rotation.
/// O(d log d) complexity. Pads input to next power of 2 if needed.
/// Uses <see cref="Vector{T}"/> for automatic ISA selection (NEON/SSE/AVX2/AVX-512).
/// </summary>
public sealed class HadamardRotation : IRotation
{
    private readonly int _originalDim;
    private readonly int _paddedDim;
    private readonly float[] _signs;
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
    /// Applies sign flips then in-place WHT, then scales.
    /// </summary>
    [SkipLocalsInit]
    public void Transform(Span<float> vector)
    {
        Guard.VectorLength(vector.Length, _paddedDim);
        ApplyElementwise(vector, _signs, static (v, s) => v * s);
        WhtInPlace(vector[.._paddedDim]);
        ApplyScalar(vector, _scale);
    }

    /// <summary>
    /// Inverse transform: WHT is self-inverse up to scaling.
    /// </summary>
    [SkipLocalsInit]
    public void InverseTransform(Span<float> vector)
    {
        Guard.VectorLength(vector.Length, _paddedDim);
        WhtInPlace(vector[.._paddedDim]);
        ApplyScalar(vector, _scale);
        ApplyElementwise(vector, _signs, static (v, s) => v * s);
    }

    /// <summary>
    /// Applies an elementwise binary operation using SIMD, with scalar tail.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyElementwise(Span<float> vector, float[] other,
        Func<Vector<float>, Vector<float>, Vector<float>> vecOp)
    {
        var i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            var w = Vector<float>.Count;
            unsafe
            {
                fixed (float* pVec = vector, pOther = other)
                {
                    for (; i + w <= _paddedDim; i += w)
                    {
                        var result = vecOp(*(Vector<float>*)(pVec + i), *(Vector<float>*)(pOther + i));
                        *(Vector<float>*)(pVec + i) = result;
                    }
                }
            }
        }

        for (; i < _paddedDim; i++)
            vector[i] *= other[i];
    }

    /// <summary>
    /// Multiplies all elements by a scalar using SIMD.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyScalar(Span<float> vector, float scalar)
    {
        var i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            var w = Vector<float>.Count;
            var s = new Vector<float>(scalar);
            unsafe
            {
                fixed (float* p = vector)
                {
                    for (; i + w <= _paddedDim; i += w)
                        *(Vector<float>*)(p + i) *= s;
                }
            }
        }

        for (; i < _paddedDim; i++)
            vector[i] *= scalar;
    }

    /// <summary>
    /// SIMD-accelerated iterative WHT butterfly, O(n log n).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WhtInPlace(Span<float> data)
    {
        var n = data.Length;
        var w = Vector.IsHardwareAccelerated ? Vector<float>.Count : 0;

        unsafe
        {
            fixed (float* p = data)
            {
                for (var halfSize = 1; halfSize < n; halfSize <<= 1)
                {
                    var stride = halfSize << 1;

                    if (halfSize >= w && w > 0)
                    {
                        for (var i = 0; i < n; i += stride)
                            for (var j = i; j + w <= i + halfSize; j += w)
                            {
                                var a = *(Vector<float>*)(p + j);
                                var b = *(Vector<float>*)(p + j + halfSize);
                                *(Vector<float>*)(p + j) = a + b;
                                *(Vector<float>*)(p + j + halfSize) = a - b;
                            }
                    }
                    else
                    {
                        for (var i = 0; i < n; i += stride)
                            for (var j = i; j < i + halfSize; j++)
                            {
                                var a = p[j];
                                var b = p[j + halfSize];
                                p[j] = a + b;
                                p[j + halfSize] = a - b;
                            }
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int NextPowerOfTwo(int v) =>
        (int)BitOperations.RoundUpToPowerOf2((uint)v);
}
