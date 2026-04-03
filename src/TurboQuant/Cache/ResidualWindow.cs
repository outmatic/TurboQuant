using TurboQuant.Core.Packing;
using TurboQuant.Core.Quantizers;

namespace TurboQuant.Cache;

/// <summary>
/// Maintains the most recent N vectors in uncompressed float32.
/// When a vector exits the window, it is compressed and moved to the associated <see cref="KVCache"/>.
/// </summary>
public sealed class ResidualWindow
{
    private readonly int _windowSize;
    private readonly int _dim;
    private readonly Queue<(float[] Key, float[] Value)> _window;
    private readonly KVCache _cache;

    /// <summary>Current number of uncompressed vectors in the window.</summary>
    public int WindowCount => _window.Count;

    /// <summary>Total entries (compressed + windowed).</summary>
    public int TotalCount => _cache.Count + _window.Count;

    /// <summary>The underlying compressed cache.</summary>
    public KVCache CompressedCache => _cache;

    /// <param name="cache">The KV cache to flush evicted vectors into.</param>
    /// <param name="windowSize">Number of recent tokens to keep uncompressed (default 128).</param>
    public ResidualWindow(KVCache cache, int windowSize = 128)
    {
        _cache = cache;
        _windowSize = windowSize;
        _dim = cache.Dimension;
        _window = new(windowSize + 1);
    }

    /// <summary>
    /// Adds a new KV pair. If the window is full, the oldest entry is compressed and moved to cache.
    /// </summary>
    public void Append(ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (_window.Count >= _windowSize)
        {
            // Evict oldest to compressed cache
            var (oldKey, oldValue) = _window.Dequeue();
            _cache.Append(oldKey, oldValue);
        }

        _window.Enqueue((key.ToArray(), value.ToArray()));
    }

    /// <summary>
    /// Gets all keys: decompressed from cache + uncompressed from window.
    /// </summary>
    public float[][] GetAllKeys() =>
        CollectAll(i => _cache.GetDecompressedKey(i), entry => entry.Key);

    /// <summary>
    /// Gets all values: decompressed from cache + uncompressed from window.
    /// </summary>
    public float[][] GetAllValues() =>
        CollectAll(i => _cache.GetDecompressedValue(i), entry => entry.Value);

    private float[][] CollectAll(Func<int, float[]> fromCache, Func<(float[] Key, float[] Value), float[]> fromWindow)
    {
        var result = new float[TotalCount][];
        var idx = 0;

        for (var i = 0; i < _cache.Count; i++)
            result[idx++] = fromCache(i);

        foreach (var entry in _window)
            result[idx++] = fromWindow(entry);

        return result;
    }

    /// <summary>Clears both the window and the compressed cache.</summary>
    public void Clear()
    {
        _window.Clear();
        _cache.Clear();
    }
}
