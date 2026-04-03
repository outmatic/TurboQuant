using System.Diagnostics;
using System.Numerics;
using TurboQuant.Cache;
using TurboQuant.Core;
using TurboQuant.Core.Quantizers;
using TurboQuant.Core.Rotation;

namespace TurboQuant;

/// <summary>
/// Fluent builder for constructing TurboQuant quantizers and caches.
/// </summary>
/// <example>
/// <code>
/// // Default: random orthogonal rotation (paper-correct, any dimension)
/// var quantizer = TurboQuantBuilder
///     .Create(dim: 768)
///     .WithBits(4)
///     .BuildMSE();
///
/// // Fast path for power-of-2 dimensions (O(d log d) instead of O(d²)):
/// var fast = TurboQuantBuilder
///     .Create(dim: 1024)
///     .WithBits(4)
///     .WithHadamardRotation()
///     .BuildMSE();
/// </code>
/// </example>
public sealed class TurboQuantBuilder
{
    private readonly int _dim;
    private int _bits = 4;
    private int _seed = 42;
    private int _keyBits = 4;
    private int _valueBits = 2;
    private int _windowSize = 128;
    private bool _useHadamard;

    private TurboQuantBuilder(int dim) => _dim = Guard.Dimension(dim);

    /// <summary>Creates a new builder for the given vector dimension.</summary>
    public static TurboQuantBuilder Create(int dim) => new(dim);

    /// <summary>Sets the quantization bit width (2, 3, or 4). Default: 4.</summary>
    public TurboQuantBuilder WithBits(int bits) { _bits = Guard.Bits(bits); return this; }

    /// <summary>Sets the random seed for deterministic rotation. Default: 42.</summary>
    public TurboQuantBuilder WithSeed(int seed) { _seed = seed; return this; }

    /// <summary>
    /// Use Walsh-Hadamard rotation. O(d log d) time, O(d) memory.
    /// Best for power-of-2 dimensions (no padding needed). For non-power-of-2
    /// dimensions, the vector is padded internally, increasing packed size by up to 33%.
    /// </summary>
    public TurboQuantBuilder WithHadamardRotation()
    {
        if (!BitOperations.IsPow2(_dim))
            Trace.TraceWarning(
                $"[TurboQuant] WithHadamardRotation() on dim={_dim} (not power-of-2) " +
                $"pads to {(int)BitOperations.RoundUpToPowerOf2((uint)_dim)}, increasing packed size. " +
                "Consider WithRandomRotation() for optimal storage, or use a power-of-2 dimension.");
        _useHadamard = true;
        return this;
    }

    /// <summary>
    /// Use paper-correct random orthogonal rotation (default).
    /// O(d²) time and memory. Works with any dimension without padding overhead.
    /// </summary>
    public TurboQuantBuilder WithRandomRotation() { _useHadamard = false; return this; }

    /// <summary>Sets the key bit width for asymmetric quantization. Default: 4.</summary>
    public TurboQuantBuilder WithKeyBits(int bits) { _keyBits = Guard.Bits(bits); return this; }

    /// <summary>Sets the value bit width for asymmetric quantization. Default: 2.</summary>
    public TurboQuantBuilder WithValueBits(int bits) { _valueBits = Guard.Bits(bits); return this; }

    /// <summary>Sets the residual window size (tokens kept uncompressed). Default: 128.</summary>
    public TurboQuantBuilder WithResidualWindow(int tokens)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tokens);
        _windowSize = tokens;
        return this;
    }

    private IRotation CreateRotation(int seed) =>
        _useHadamard ? new HadamardRotation(_dim, seed) : new RandomRotation(_dim, seed);

    /// <summary>Builds an MSE-optimal vector quantizer.</summary>
    public TurboQuantMSE BuildMSE() => new(_dim, _bits, CreateRotation(_seed));

    /// <summary>Builds an asymmetric quantizer for KV cache compression.</summary>
    public TurboQuantAsymmetric BuildAsymmetric() =>
        new(_dim, _keyBits, _valueBits, CreateRotation(_seed), CreateRotation(_seed + 1));

    /// <summary>Builds a complete KV cache with residual window.</summary>
    public (KVCache Cache, ResidualWindow Window) BuildKVCache()
    {
        var cache = new KVCache(_dim, _keyBits, _valueBits, _seed);
        return (cache, new ResidualWindow(cache, _windowSize));
    }
}
