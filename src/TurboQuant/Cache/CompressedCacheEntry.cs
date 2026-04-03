using TurboQuant.Core.Packing;

namespace TurboQuant.Cache;

/// <summary>
/// A single compressed KV pair in the cache.
/// </summary>
public readonly record struct CompressedCacheEntry(
    PackedVector Key,
    PackedVector Value);
