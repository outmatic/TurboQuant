using System.Runtime.CompilerServices;

namespace TurboQuant.Core;

/// <summary>
/// Shared math utilities used across the library.
/// </summary>
internal static class MathUtils
{
    /// <summary>
    /// Generates a standard normal random variate using the Box-Muller transform.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double NextGaussian(Random rng)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
