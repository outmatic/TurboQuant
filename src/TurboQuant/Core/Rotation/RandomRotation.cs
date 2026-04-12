using System.Buffers;
using System.Runtime.CompilerServices;
using TurboQuant.Core.Simd;

namespace TurboQuant.Core.Rotation;

/// <summary>
/// Random orthogonal rotation via QR decomposition of a Gaussian random matrix.
/// This is the rotation specified by the TurboQuant paper (Zandieh et al., ICLR 2026):
/// P is drawn uniformly from the orthogonal group O(d) by QR factorization of a
/// matrix with i.i.d. N(0,1) entries.
/// </summary>
/// <remarks>
/// O(d^2) transform and O(d^2) storage. For a faster O(d log d) approximation,
/// use <see cref="HadamardRotation"/> (not paper-correct but practically similar).
/// </remarks>
public sealed class RandomRotation : IRotation
{
    private readonly int _dim;
    private readonly float[] _q;  // orthogonal matrix, row-major, dim x dim
    private readonly float[] _qt; // transpose, row-major, dim x dim

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

        for (var i = 0; i < _dim * _dim; i++)
            _q[i] = (float)MathUtils.NextGaussian(rng);

        // QR via modified Gram-Schmidt
        for (var i = 0; i < _dim; i++)
        {
            var rowI = i * _dim;

            for (var j = 0; j < i; j++)
            {
                var rowJ = j * _dim;
                var dot = DotProductSimd.DotProduct(
                    _q.AsSpan(rowI, _dim), _q.AsSpan(rowJ, _dim));
                for (var k = 0; k < _dim; k++)
                    _q[rowI + k] -= dot * _q[rowJ + k];
            }

            var norm = DotProductSimd.L2Norm(_q.AsSpan(rowI, _dim));
            if (norm < 1e-10f)
                throw new InvalidOperationException("Degenerate random matrix.");

            var invNorm = 1f / norm;
            for (var k = 0; k < _dim; k++)
                _q[rowI + k] *= invNorm;
        }

        for (var i = 0; i < _dim; i++)
            for (var j = 0; j < _dim; j++)
                _qt[i * _dim + j] = _q[j * _dim + i];
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
    /// Matrix-vector multiply: out[i] = dot(matrix[i,:], vec).
    /// Delegates per-row dot product to <see cref="DotProductSimd"/>.
    /// </summary>
    private void MatVecMultiply(float[] matrix, Span<float> vec, Span<float> output)
    {
        ReadOnlySpan<float> v = vec[.._dim];
        for (var i = 0; i < _dim; i++)
            output[i] = DotProductSimd.DotProduct(matrix.AsSpan(i * _dim, _dim), v);
    }
}
