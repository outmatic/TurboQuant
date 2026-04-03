using TurboQuant.Core.Packing;
using TurboQuant.Diagnostics;

namespace TurboQuant.Core.Quantizers;

/// <summary>
/// Defines the contract for a vector quantizer.
/// </summary>
public interface IVectorQuantizer
{
    /// <summary>
    /// Original (unpadded) vector dimension.
    /// </summary>
    int Dimension { get; }

    /// <summary>
    /// Number of quantization bits per coordinate.
    /// </summary>
    int Bits { get; }

    /// <summary>
    /// Quantizes a single vector.
    /// </summary>
    PackedVector Quantize(ReadOnlySpan<float> vector);

    /// <summary>
    /// Dequantizes a packed vector back to float representation.
    /// </summary>
    float[] Dequantize(PackedVector packed);

    /// <summary>
    /// Quantizes a batch of vectors with zero heap allocation.
    /// Vectors are contiguous in <paramref name="vectors"/>: each is <see cref="Dimension"/> floats wide.
    /// </summary>
    void QuantizeBatch(ReadOnlySpan<float> vectors, Span<PackedVector> output);

    /// <summary>
    /// Computes approximate similarity between two packed vectors without full decompression.
    /// </summary>
    float ApproxSimilarity(PackedVector a, PackedVector b);

    /// <summary>
    /// Returns runtime compression statistics.
    /// </summary>
    CompressionStats GetStats();
}
