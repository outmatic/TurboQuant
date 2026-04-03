using TurboQuant.Core.Packing;
using TurboQuant.Core.Quantizers;

namespace TurboQuant.Cache;

/// <summary>
/// Compressed KV cache for LLM inference. Stores key-value pairs using TurboQuant compression.
/// </summary>
public sealed class KVCache
{
    private readonly TurboQuantAsymmetric _quantizer;
    private readonly List<CompressedCacheEntry> _entries;

    /// <summary>Number of cached entries.</summary>
    public int Count => _entries.Count;

    /// <summary>Vector dimension.</summary>
    public int Dimension => _quantizer.Dimension;

    /// <param name="dimension">Vector dimension (model head dim).</param>
    /// <param name="keyBits">Bit width for keys (default 4).</param>
    /// <param name="valueBits">Bit width for values (default 2).</param>
    /// <param name="seed">Random seed.</param>
    /// <param name="initialCapacity">Initial list capacity.</param>
    public KVCache(int dimension, int keyBits = 4, int valueBits = 2, int seed = 42, int initialCapacity = 256)
    {
        _quantizer = new(dimension, keyBits, valueBits, seed);
        _entries = new(initialCapacity);
    }

    /// <summary>
    /// Appends a key-value pair, compressing both vectors.
    /// </summary>
    public void Append(ReadOnlySpan<float> keyVector, ReadOnlySpan<float> valueVector)
    {
        var compressedKey = _quantizer.QuantizeKey(keyVector);
        var compressedValue = _quantizer.QuantizeValue(valueVector);
        _entries.Add(new(compressedKey, compressedValue));
    }

    /// <summary>Returns all compressed keys.</summary>
    public IReadOnlyList<PackedVector> GetKeys()
    {
        var keys = new PackedVector[_entries.Count];
        for (int i = 0; i < _entries.Count; i++)
            keys[i] = _entries[i].Key;
        return keys;
    }

    /// <summary>Returns all compressed values.</summary>
    public IReadOnlyList<PackedVector> GetValues()
    {
        var values = new PackedVector[_entries.Count];
        for (int i = 0; i < _entries.Count; i++)
            values[i] = _entries[i].Value;
        return values;
    }

    /// <summary>Dequantizes and returns the key at the given index.</summary>
    public float[] GetDecompressedKey(int index) => _quantizer.DequantizeKey(_entries[index].Key);

    /// <summary>Dequantizes and returns the value at the given index.</summary>
    public float[] GetDecompressedValue(int index) => _quantizer.DequantizeValue(_entries[index].Value);

    /// <summary>Removes all entries.</summary>
    public void Clear()
    {
        _entries.Clear();
    }

    /// <summary>Trims the cache to keep only the most recent <paramref name="maxTokens"/> entries.</summary>
    public void TrimToWindow(int maxTokens)
    {
        if (_entries.Count > maxTokens)
        {
            _entries.RemoveRange(0, _entries.Count - maxTokens);
        }
    }
}
