namespace TurboQuant.Core.Rotation;

/// <summary>
/// Defines a rotation transform for vector preprocessing before quantization.
/// </summary>
public interface IRotation
{
    /// <summary>
    /// Transforms the input vector in-place.
    /// </summary>
    void Transform(Span<float> vector);

    /// <summary>
    /// Applies the inverse transform in-place.
    /// </summary>
    void InverseTransform(Span<float> vector);

    /// <summary>
    /// The padded dimension (power of 2) used by this rotation.
    /// </summary>
    int PaddedDimension { get; }
}
