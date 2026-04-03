using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using TurboQuant.Core.Codebook;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Rotation;
using TurboQuant.Core.Simd;
using TurboQuant.Diagnostics;

namespace TurboQuant.Core.Quantizers;

/// <summary>
/// TurboQuant MSE-optimal vector quantizer. Implements Algorithm 1 from
/// "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
/// (Zandieh et al., ICLR 2026).
/// </summary>
/// <remarks>
/// <para>
/// Pipeline: normalize → random orthogonal rotation Π → per-coordinate scalar
/// quantization with Beta(d/2,1/2) Lloyd-Max codebook → bit-pack.
/// </para>
/// <para>
/// The random rotation Π is drawn uniformly from O(d) via QR decomposition of a
/// Gaussian matrix (as specified in the paper). Each coordinate of Π·x follows
/// the exact Beta distribution, enabling near-optimal distortion D_mse = d·C(f_X,b).
/// </para>
/// <para>
/// Does NOT use QJL correction (confirmed to degrade performance with softmax attention).
/// </para>
/// </remarks>
public sealed class TurboQuantMSE : IVectorQuantizer
{
    private readonly IRotation _rotation;
    private readonly BetaCodebook _codebook;
    private readonly int _dim;
    private readonly int _rotDim; // = rotation.PaddedDimension (= dim for RandomRotation)
    private readonly int _bits;
    private readonly int _packedBytes;
    private readonly CompressionStats _stats;

    /// <inheritdoc/>
    public int Dimension => _dim;

    /// <inheritdoc/>
    public int Bits => _bits;

    /// <summary>
    /// Effective dimension used by the rotation (equals <see cref="Dimension"/> for
    /// the paper-correct random orthogonal rotation; may be larger for WHT).
    /// </summary>
    public int RotationDimension => _rotDim;

    /// <summary>
    /// Creates a quantizer using paper-correct random orthogonal rotation.
    /// For the faster Hadamard rotation (power-of-2 dimensions), use
    /// <see cref="TurboQuantMSE(int, int, IRotation)"/> with <see cref="HadamardRotation"/>,
    /// or the builder with <c>.WithHadamardRotation()</c>.
    /// </summary>
    /// <param name="dimension">Original vector dimension (e.g. 768, 1024).</param>
    /// <param name="bits">Quantization bits per coordinate (2, 3, or 4).</param>
    /// <param name="seed">Seed for deterministic rotation.</param>
    public TurboQuantMSE(int dimension, int bits, int seed = 42)
        : this(dimension, bits, new RandomRotation(dimension, seed))
    {
    }

    /// <summary>
    /// Creates a quantizer with a custom rotation.
    /// </summary>
    /// <param name="dimension">Original vector dimension.</param>
    /// <param name="bits">Quantization bits per coordinate (2, 3, or 4).</param>
    /// <param name="rotation">Rotation transform to use.</param>
    public TurboQuantMSE(int dimension, int bits, IRotation rotation)
    {
        _dim = Guard.Dimension(dimension);
        _bits = Guard.Bits(bits);
        _rotation = rotation;
        _rotDim = rotation.PaddedDimension;
        _codebook = CodebookCache.GetOrCreate(bits, _rotDim);
        _packedBytes = PackedVector.GetPackedByteCount(_rotDim, bits);
        _stats = new(bits, dimension);
    }

    /// <inheritdoc/>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public PackedVector Quantize(ReadOnlySpan<float> vector)
    {
        Guard.VectorLength(vector.Length, _dim);

        var startTicks = Stopwatch.GetTimestamp();

        var buffer = ArrayPool<float>.Shared.Rent(_rotDim);
        var indices = ArrayPool<int>.Shared.Rent(_rotDim);

        try
        {
            var buf = buffer.AsSpan(0, _rotDim);

            // Copy input and pad if needed (WHT path)
            vector[.._dim].CopyTo(buf);
            if (_rotDim > _dim) buf[_dim..].Clear();

            // Step 1: Compute and store L2 norm (paper §3: "store norms in floating-point")
            var norm = DotProductSimd.L2Norm(buf[.._dim]);
            if (norm < 1e-10f)
                return new(new byte[_packedBytes], 0f, _dim, _bits, _rotDim);

            // Step 2: Normalize to unit sphere S^{d-1}
            var invNorm = 1f / norm;
            for (int i = 0; i < _dim; i++)
                buf[i] *= invNorm;

            // Step 3: y = Π · x  (random orthogonal rotation)
            _rotation.Transform(buf);

            // Step 4: idx_j = argmin_k |y_j - c_k|  (per-coordinate quantization)
            var idx = indices.AsSpan(0, _rotDim);
            for (int i = 0; i < _rotDim; i++)
                idx[i] = _codebook.Quantize(buf[i]);

            // Step 5: Bit-pack indices
            var data = new byte[_packedBytes];
            BitPacker.Pack(idx, data, _bits);

            // Record D_mse = ||x_unit - x̃_unit||² = Σ(y_j - c_{idx_j})²
            // (rotation preserves norms, so this equals the reconstruction error)
            double dmse = 0;
            for (int i = 0; i < _rotDim; i++)
            {
                double diff = buf[i] - _codebook.Dequantize(idx[i]);
                dmse += diff * diff;
            }

            var elapsed = Stopwatch.GetTimestamp() - startTicks;
            _stats.RecordQuantize(elapsed, dmse);

            return new(data, norm, _dim, _bits, _rotDim);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
            ArrayPool<int>.Shared.Return(indices);
        }
    }

    /// <inheritdoc/>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public float[] Dequantize(PackedVector packed)
    {
        var result = new float[_dim];
        Dequantize(packed, result);
        return result;
    }

    /// <summary>
    /// Dequantizes into an existing buffer (zero allocation).
    /// </summary>
    /// <param name="packed">The packed vector to decompress.</param>
    /// <param name="output">Destination span, must be at least <see cref="Dimension"/> elements.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Dequantize(PackedVector packed, Span<float> output)
    {
        Guard.VectorLength(output.Length, _dim);
        var rotDim = packed.PaddedDim;
        var buffer = ArrayPool<float>.Shared.Rent(rotDim);
        var indexes = ArrayPool<int>.Shared.Rent(rotDim);

        try
        {
            var buf = buffer.AsSpan(0, rotDim);
            var idx = indexes.AsSpan(0, rotDim);

            BitPacker.Unpack(packed.Data.Span, idx, rotDim, _bits);

            for (var i = 0; i < rotDim; i++)
                buf[i] = _codebook.Dequantize(idx[i]);

            _rotation.InverseTransform(buf);

            for (var i = 0; i < _dim; i++)
                output[i] = buf[i] * packed.Norm;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
            ArrayPool<int>.Shared.Return(indexes);
        }
    }

    /// <inheritdoc/>
    public void QuantizeBatch(ReadOnlySpan<float> vectors, Span<PackedVector> output)
    {
        var count = vectors.Length / _dim;
        ArgumentOutOfRangeException.ThrowIfLessThan(output.Length, count, nameof(output));

        for (int i = 0; i < count; i++)
            output[i] = Quantize(vectors.Slice(i * _dim, _dim));
    }

    /// <inheritdoc/>
    [SkipLocalsInit]
    public float ApproxSimilarity(PackedVector a, PackedVector b)
    {
        if (a.Bits != b.Bits || a.PaddedDim != b.PaddedDim)
            throw new ArgumentException("PackedVectors must have matching bits and dimensions.");

        return _bits == 4
            ? ApproxSimilarity4Bit(a.Data.Span, b.Data.Span)
            : ApproxSimilarityGeneric(a, b);
    }

    /// <summary>
    /// Fast path for 4-bit: reads index pairs directly from packed bytes
    /// and resolves dot products via precomputed LUT. No unpacking, no centroid lookup.
    /// </summary>
    private float ApproxSimilarity4Bit(ReadOnlySpan<byte> dataA, ReadOnlySpan<byte> dataB)
    {
        var dot = 0f;
        var normA2 = 0f;
        var normB2 = 0f;

        for (var i = 0; i < dataA.Length; i++)
        {
            var ba = dataA[i];
            var bb = dataB[i];

            // Each byte holds 2 indices: low nibble and high nibble
            var a0 = ba & 0xF;
            var a1 = (ba >> 4) & 0xF;
            var b0 = bb & 0xF;
            var b1 = (bb >> 4) & 0xF;

            dot += _codebook.DotLookup(a0, b0) + _codebook.DotLookup(a1, b1);
            normA2 += _codebook.NormSqLookup(a0) + _codebook.NormSqLookup(a1);
            normB2 += _codebook.NormSqLookup(b0) + _codebook.NormSqLookup(b1);
        }

        var denom = MathF.Sqrt(normA2 * normB2);
        return denom > 1e-10f ? dot / denom : 0f;
    }

    /// <summary>Fallback for 2/3-bit: unpack then lookup.</summary>
    private float ApproxSimilarityGeneric(PackedVector a, PackedVector b)
    {
        var rotDim = a.PaddedDim;
        var idxA = ArrayPool<int>.Shared.Rent(rotDim);
        var idxB = ArrayPool<int>.Shared.Rent(rotDim);

        try
        {
            BitPacker.Unpack(a.Data.Span, idxA.AsSpan(0, rotDim), rotDim, _bits);
            BitPacker.Unpack(b.Data.Span, idxB.AsSpan(0, rotDim), rotDim, _bits);

            var dot = 0f;
            var normA2 = 0f;
            var normB2 = 0f;

            for (var i = 0; i < rotDim; i++)
            {
                dot += _codebook.DotLookup(idxA[i], idxB[i]);
                normA2 += _codebook.NormSqLookup(idxA[i]);
                normB2 += _codebook.NormSqLookup(idxB[i]);
            }

            var denom = MathF.Sqrt(normA2 * normB2);
            return denom > 1e-10f ? dot / denom : 0f;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(idxA);
            ArrayPool<int>.Shared.Return(idxB);
        }
    }

    /// <inheritdoc/>
    public CompressionStats GetStats() => _stats;
}
