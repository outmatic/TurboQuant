using TurboQuant.Cache;
using TurboQuant.Core.Quantizers;
using Xunit;

namespace TurboQuant.Tests.Integration;

public class KVCacheIntegrationTests
{
    [Fact]
    public void KVCache_AppendAndRetrieve()
    {
        int dim = 256;
        var cache = new KVCache(dim, keyBits: 4, valueBits: 2, seed: 42);
        var rng = new Random(123); // different seed from quantizer to avoid correlation

        var key = GenerateRandom(dim, rng, normalize: true);
        var value = GenerateRandom(dim, rng, normalize: true);

        cache.Append(key, value);
        Assert.Equal(1, cache.Count);

        var keys = cache.GetKeys();
        Assert.Single(keys);

        var decompressedKey = cache.GetDecompressedKey(0);
        float cosine = CosineSim(key, decompressedKey);
        Assert.True(cosine > 0.95f, $"Key cosine {cosine} too low");
    }

    [Fact]
    public void KVCache_TrimToWindow()
    {
        int dim = 32;
        var cache = new KVCache(dim, seed: 42);
        var rng = new Random(42);

        for (int i = 0; i < 100; i++)
            cache.Append(GenerateRandom(dim, rng), GenerateRandom(dim, rng));

        Assert.Equal(100, cache.Count);
        cache.TrimToWindow(50);
        Assert.Equal(50, cache.Count);
    }

    [Fact]
    public void ResidualWindow_EvictsCorrectly()
    {
        int dim = 32;
        var cache = new KVCache(dim, seed: 42);
        var window = new ResidualWindow(cache, windowSize: 4);
        var rng = new Random(42);

        // Add 6 entries: first 2 should be evicted to cache
        for (int i = 0; i < 6; i++)
            window.Append(GenerateRandom(dim, rng), GenerateRandom(dim, rng));

        Assert.Equal(2, cache.Count);       // 2 evicted
        Assert.Equal(4, window.WindowCount); // 4 in window
        Assert.Equal(6, window.TotalCount);
    }

    [Fact]
    public void Builder_FluentAPI_Works()
    {
        var quantizer = TurboQuantBuilder
            .Create(dim: 128)
            .WithBits(4)
            .WithSeed(42)
            .BuildMSE();

        Assert.Equal(128, quantizer.Dimension);
        Assert.Equal(4, quantizer.Bits);

        var rng = new Random(42);
        var v = GenerateRandom(128, rng);
        var packed = quantizer.Quantize(v);
        var restored = quantizer.Dequantize(packed);
        Assert.Equal(128, restored.Length);
    }

    [Fact]
    public void Builder_KVCache_Works()
    {
        var (cache, window) = TurboQuantBuilder
            .Create(dim: 64)
            .WithKeyBits(4)
            .WithValueBits(2)
            .WithResidualWindow(tokens: 16)
            .BuildKVCache();

        Assert.Equal(64, cache.Dimension);
    }

    private static float[] GenerateRandom(int dim, Random rng, bool normalize = false)
    {
        var v = new float[dim];
        float norm = 0;
        for (int i = 0; i < dim; i++)
        {
            v[i] = (float)(rng.NextDouble() * 2 - 1);
            norm += v[i] * v[i];
        }
        if (normalize)
        {
            norm = MathF.Sqrt(norm);
            for (int i = 0; i < dim; i++)
                v[i] /= norm;
        }
        return v;
    }

    private static float CosineSim(float[] a, float[] b)
    {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / MathF.Sqrt(na * nb);
    }
}
