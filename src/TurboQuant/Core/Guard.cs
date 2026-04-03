using System.Runtime.CompilerServices;

namespace TurboQuant.Core;

/// <summary>
/// Shared argument validation helpers.
/// </summary>
internal static class Guard
{
    internal static int Dimension(int dimension, [CallerArgumentExpression(nameof(dimension))] string? name = null) =>
        dimension > 0 ? dimension : throw new ArgumentOutOfRangeException(name, "Dimension must be positive.");

    internal static int Bits(int bits, [CallerArgumentExpression(nameof(bits))] string? name = null) =>
        bits is >= 2 and <= 4 ? bits : throw new ArgumentOutOfRangeException(name, "Supported bit widths: 2, 3, 4.");

    internal static void VectorLength(int length, int required, [CallerArgumentExpression(nameof(length))] string? name = null)
    {
        if (length < required)
            throw new ArgumentException($"Vector length {length} < required {required}.", name);
    }
}
