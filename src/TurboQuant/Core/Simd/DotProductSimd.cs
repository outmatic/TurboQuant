using System.Numerics;
using System.Runtime.CompilerServices;

namespace TurboQuant.Core.Simd;

/// <summary>
/// SIMD-accelerated dot product, cosine similarity, and L2 norm.
/// Uses <see cref="Vector{T}"/> for automatic ISA selection (NEON/SSE/AVX2/AVX-512).
/// Dual accumulators hide data-dependency latency.
/// </summary>
public static class DotProductSimd
{
    /// <summary>
    /// Computes dot product of two float spans.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var n = Math.Min(a.Length, b.Length);
        var sum = 0f;
        var i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            var w = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;

            unsafe
            {
                fixed (float* pA = a, pB = b)
                {
                    for (; i + w * 2 <= n; i += w * 2)
                    {
                        acc0 += *(Vector<float>*)(pA + i) * *(Vector<float>*)(pB + i);
                        acc1 += *(Vector<float>*)(pA + i + w) * *(Vector<float>*)(pB + i + w);
                    }
                    for (; i + w <= n; i += w)
                        acc0 += *(Vector<float>*)(pA + i) * *(Vector<float>*)(pB + i);
                }
            }

            sum = Vector.Sum(acc0 + acc1);
        }

        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>
    /// Computes cosine similarity between two float spans.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var n = Math.Min(a.Length, b.Length);
        var dot = 0f;
        var normA = 0f;
        var normB = 0f;
        var i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            var w = Vector<float>.Count;
            var accD = Vector<float>.Zero;
            var accA = Vector<float>.Zero;
            var accB = Vector<float>.Zero;

            unsafe
            {
                fixed (float* pA = a, pB = b)
                {
                    for (; i + w <= n; i += w)
                    {
                        var va = *(Vector<float>*)(pA + i);
                        var vb = *(Vector<float>*)(pB + i);
                        accD += va * vb;
                        accA += va * va;
                        accB += vb * vb;
                    }
                }
            }

            dot = Vector.Sum(accD);
            normA = Vector.Sum(accA);
            normB = Vector.Sum(accB);
        }

        for (; i < n; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        var denom = MathF.Sqrt(normA * normB);
        return denom > 1e-10f ? dot / denom : 0f;
    }

    /// <summary>
    /// Computes L2 norm of a float span.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float L2Norm(ReadOnlySpan<float> vector)
    {
        var sum = 0f;
        var i = 0;

        if (Vector.IsHardwareAccelerated)
        {
            var w = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;

            unsafe
            {
                fixed (float* p = vector)
                {
                    for (; i + w * 2 <= vector.Length; i += w * 2)
                    {
                        var v0 = *(Vector<float>*)(p + i);
                        var v1 = *(Vector<float>*)(p + i + w);
                        acc0 += v0 * v0;
                        acc1 += v1 * v1;
                    }
                    for (; i + w <= vector.Length; i += w)
                    {
                        var v0 = *(Vector<float>*)(p + i);
                        acc0 += v0 * v0;
                    }
                }
            }

            sum = Vector.Sum(acc0 + acc1);
        }

        for (; i < vector.Length; i++)
            sum += vector[i] * vector[i];

        return MathF.Sqrt(sum);
    }
}
