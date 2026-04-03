using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TurboQuant.Core.Simd;

namespace TurboQuant.Core.Rotation;

/// <summary>
/// Random orthogonal rotation via QR decomposition of a Gaussian random matrix.
/// This is the rotation specified by the TurboQuant paper (Zandieh et al., ICLR 2026):
/// Π is drawn uniformly from the orthogonal group O(d) by QR factorization of a
/// matrix with i.i.d. N(0,1) entries.
/// </summary>
/// <remarks>
/// O(d²) transform and O(d²) storage. For a faster O(d log d) approximation,
/// use <see cref="HadamardRotation"/> (not paper-correct but practically similar).
/// </remarks>
public sealed class RandomRotation : IRotation
{
    private readonly int _dim;
    private readonly float[] _q;  // orthogonal matrix, row-major, dim × dim
    private readonly float[] _qt; // transpose, row-major, dim × dim

    /// <inheritdoc/>
    public int PaddedDimension => _dim;

    /// <param name="dimension">Vector dimension.</param>
    /// <param name="seed">Random seed for reproducible rotation matrix.</param>
    public RandomRotation(int dimension, int seed = 42)
    {
        _dim = Guard.Dimension(dimension);
        _q = new float[_dim * _dim];
        _qt = new float[_dim * _dim];

        var rng = new Random(seed);

        // Fill with i.i.d. N(0,1) via Box-Muller
        for (var i = 0; i < _dim * _dim; i++)
            _q[i] = (float)NextGaussian(rng);

        // QR via modified Gram-Schmidt (numerically stable)
        for (var i = 0; i < _dim; i++)
        {
            var rowI = i * _dim;

            // Subtract projections onto all previous rows
            for (var j = 0; j < i; j++)
            {
                var rowJ = j * _dim;
                var dot = DotRow(rowI, rowJ);
                for (var k = 0; k < _dim; k++)
                    _q[rowI + k] -= dot * _q[rowJ + k];
            }

            // Normalize
            var norm = 0f;
            for (var k = 0; k < _dim; k++)
                norm += _q[rowI + k] * _q[rowI + k];
            norm = MathF.Sqrt(norm);

            if (norm < 1e-10f)
                throw new InvalidOperationException("Degenerate random matrix.");

            var invNorm = 1f / norm;
            for (var k = 0; k < _dim; k++)
                _q[rowI + k] *= invNorm;
        }

        // Compute transpose
        for (var i = 0; i < _dim; i++)
            for (var j = 0; j < _dim; j++)
                _qt[i * _dim + j] = _q[j * _dim + i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float DotRow(int rowOffsetA, int rowOffsetB)
    {
        var sum = 0f;
        for (var k = 0; k < _dim; k++)
            sum += _q[rowOffsetA + k] * _q[rowOffsetB + k];
        return sum;
    }

    /// <inheritdoc/>
    [SkipLocalsInit]
    public void Transform(Span<float> vector)
    {
        Guard.VectorLength(vector.Length, _dim);

        var temp = ArrayPool<float>.Shared.Rent(_dim);
        try
        {
            MatVecMultiply(_q, vector, temp.AsSpan(0, _dim));
            temp.AsSpan(0, _dim).CopyTo(vector);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(temp);
        }
    }

    /// <inheritdoc/>
    [SkipLocalsInit]
    public void InverseTransform(Span<float> vector)
    {
        Guard.VectorLength(vector.Length, _dim);

        var temp = ArrayPool<float>.Shared.Rent(_dim);
        try
        {
            MatVecMultiply(_qt, vector, temp.AsSpan(0, _dim));
            temp.AsSpan(0, _dim).CopyTo(vector);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(temp);
        }
    }

    /// <summary>
    /// SIMD-accelerated matrix-vector multiply: out[i] = dot(matrix[i,:], vec).
    /// Uses AVX2 (8-wide), SSE (4-wide), or scalar fallback.
    /// </summary>
    private void MatVecMultiply(float[] matrix, Span<float> vec, Span<float> output)
    {
        if (Avx2.IsSupported)
            MatVecAvx2(matrix, vec, output);
        else if (Sse.IsSupported)
            MatVecSse(matrix, vec, output);
        else
            MatVecScalar(matrix, vec, output);
    }

    private void MatVecAvx2(float[] matrix, Span<float> vec, Span<float> output)
    {
        unsafe
        {
            fixed (float* pMat = matrix, pVec = vec, pOut = output)
            {
                for (var i = 0; i < _dim; i++)
                {
                    var pRow = pMat + i * _dim;
                    var acc = Vector256<float>.Zero;
                    var j = 0;

                    for (; j + 8 <= _dim; j += 8)
                    {
                        var mr = Avx.LoadVector256(pRow + j);
                        var vr = Avx.LoadVector256(pVec + j);
                        acc = Avx.Add(acc, Avx.Multiply(mr, vr));
                    }

                    var sum = DotProductSimd.HorizontalSum256(acc);

                    // Scalar remainder
                    for (; j < _dim; j++)
                        sum += pRow[j] * pVec[j];

                    pOut[i] = sum;
                }
            }
        }
    }

    private void MatVecSse(float[] matrix, Span<float> vec, Span<float> output)
    {
        unsafe
        {
            fixed (float* pMat = matrix, pVec = vec, pOut = output)
            {
                for (var i = 0; i < _dim; i++)
                {
                    var pRow = pMat + i * _dim;
                    var acc = Vector128<float>.Zero;
                    var j = 0;

                    for (; j + 4 <= _dim; j += 4)
                    {
                        var mr = Sse.LoadVector128(pRow + j);
                        var vr = Sse.LoadVector128(pVec + j);
                        acc = Sse.Add(acc, Sse.Multiply(mr, vr));
                    }

                    var sum = DotProductSimd.HorizontalSum128(acc);

                    for (; j < _dim; j++)
                        sum += pRow[j] * pVec[j];

                    pOut[i] = sum;
                }
            }
        }
    }

    private void MatVecScalar(float[] matrix, Span<float> vec, Span<float> output)
    {
        for (var i = 0; i < _dim; i++)
        {
            var rowOffset = i * _dim;
            var sum = 0f;
            for (var j = 0; j < _dim; j++)
                sum += matrix[rowOffset + j] * vec[j];
            output[i] = sum;
        }
    }

    private static double NextGaussian(Random rng) => MathUtils.NextGaussian(rng);
}
