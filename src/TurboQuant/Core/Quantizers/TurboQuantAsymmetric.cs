using TurboQuant.Core.Packing;
using TurboQuant.Core.Rotation;

namespace TurboQuant.Core.Quantizers;

/// <summary>
/// Asymmetric quantizer for KV cache compression.
/// Uses different bit widths for keys (higher precision) and values (lower precision).
/// </summary>
/// <remarks>
/// Typical config: keys at 4-bit, values at 2-bit.
/// Does NOT use QJL correction. MSE-only is optimal for softmax attention.
/// </remarks>
public sealed class TurboQuantAsymmetric
{
    private readonly TurboQuantMSE _keyQuantizer;
    private readonly TurboQuantMSE _valueQuantizer;

    /// <summary>Vector dimension.</summary>
    public int Dimension => _keyQuantizer.Dimension;

    /// <summary>Bits used for key quantization.</summary>
    public int KeyBits => _keyQuantizer.Bits;

    /// <summary>Bits used for value quantization.</summary>
    public int ValueBits => _valueQuantizer.Bits;

    /// <summary>
    /// Creates an asymmetric quantizer using paper-correct random orthogonal rotation.
    /// </summary>
    /// <param name="dimension">Vector dimension (must match model head dim).</param>
    /// <param name="keyBits">Bit width for keys (2-4, typically 4).</param>
    /// <param name="valueBits">Bit width for values (2-4, typically 2).</param>
    /// <param name="seed">Seed for deterministic rotations.</param>
    public TurboQuantAsymmetric(int dimension, int keyBits = 4, int valueBits = 2, int seed = 42)
    {
        _keyQuantizer = new TurboQuantMSE(dimension, keyBits, seed);
        _valueQuantizer = new TurboQuantMSE(dimension, valueBits, seed + 1);
    }

    /// <summary>
    /// Creates an asymmetric quantizer with a custom rotation (e.g. HadamardRotation).
    /// </summary>
    public TurboQuantAsymmetric(int dimension, int keyBits, int valueBits, IRotation keyRotation, IRotation valueRotation)
    {
        _keyQuantizer = new TurboQuantMSE(dimension, keyBits, keyRotation);
        _valueQuantizer = new TurboQuantMSE(dimension, valueBits, valueRotation);
    }

    /// <summary>Quantizes a key vector.</summary>
    public PackedVector QuantizeKey(ReadOnlySpan<float> key) => _keyQuantizer.Quantize(key);

    /// <summary>Quantizes a value vector.</summary>
    public PackedVector QuantizeValue(ReadOnlySpan<float> value) => _valueQuantizer.Quantize(value);

    /// <summary>Dequantizes a key vector.</summary>
    public float[] DequantizeKey(PackedVector packed) => _keyQuantizer.Dequantize(packed);

    /// <summary>Dequantizes a value vector.</summary>
    public float[] DequantizeValue(PackedVector packed) => _valueQuantizer.Dequantize(packed);
}
