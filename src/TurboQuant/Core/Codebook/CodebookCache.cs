using System.Collections.Concurrent;

namespace TurboQuant.Core.Codebook;

/// <summary>
/// Thread-safe cache of <see cref="BetaCodebook"/> instances keyed by (bits, dimension).
/// Since the codebook depends on the vector dimension (the Beta distribution shape parameter),
/// each unique (bits, dimension) pair gets its own codebook computed once.
/// </summary>
public static class CodebookCache
{
    private static readonly ConcurrentDictionary<(int bits, int dim), BetaCodebook> _cache = new();

    /// <summary>
    /// Gets or creates a BetaCodebook for the given bit width and dimension.
    /// </summary>
    public static BetaCodebook GetOrCreate(int bits, int dimension)
        => _cache.GetOrAdd((bits, dimension), key => new BetaCodebook(key.bits, key.dim));
}
