namespace TurboQuant.Core.Codebook;

/// <summary>
/// Defines a scalar codebook for mapping continuous values to discrete levels.
/// </summary>
public interface ICodebook
{
    /// <summary>
    /// Number of quantization bits (2, 3, or 4).
    /// </summary>
    int Bits { get; }

    /// <summary>
    /// Number of quantization levels (2^Bits).
    /// </summary>
    int Levels { get; }

    /// <summary>
    /// Returns the centroid values for all levels.
    /// </summary>
    ReadOnlySpan<float> Centroids { get; }

    /// <summary>
    /// Finds the nearest centroid index for the given value.
    /// </summary>
    int Quantize(float value);

    /// <summary>
    /// Returns the centroid value for the given index.
    /// </summary>
    float Dequantize(int index);
}
