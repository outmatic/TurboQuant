using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TurboQuant.Core.Simd;

/// <summary>
/// SIMD-accelerated dot product and cosine similarity operations.
/// </summary>
public static class DotProductSimd
{
    /// <summary>
    /// Computes dot product of two float spans using the best available instruction set.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var n = Math.Min(a.Length, b.Length);

        if (Avx2.IsSupported && n >= 8)
            return DotProductAvx2(a, b, n);
        if (Sse2.IsSupported && n >= 4)
            return DotProductSse2(a, b, n);
        return DotProductScalar(a, b, n);
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

        if (Avx2.IsSupported && n >= 8)
        {
            DotAndNormsAvx2(a, b, n, out dot, out normA, out normB);
        }
        else if (Sse2.IsSupported && n >= 4)
        {
            DotAndNormsSse2(a, b, n, out dot, out normA, out normB);
        }
        else
        {
            for (var i = 0; i < n; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
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

        if (Avx2.IsSupported && vector.Length >= 8)
        {
            unsafe
            {
                fixed (float* ptr = vector)
                {
                    var acc = Vector256<float>.Zero;
                    for (; i + 8 <= vector.Length; i += 8)
                    {
                        var v = Avx.LoadVector256(ptr + i);
                        acc = Avx.Add(acc, Avx.Multiply(v, v));
                    }
                    sum = HorizontalSum256(acc);
                }
            }
        }
        else if (Sse2.IsSupported && vector.Length >= 4)
        {
            unsafe
            {
                fixed (float* ptr = vector)
                {
                    var acc = Vector128<float>.Zero;
                    for (; i + 4 <= vector.Length; i += 4)
                    {
                        var v = Sse.LoadVector128(ptr + i);
                        acc = Sse.Add(acc, Sse.Multiply(v, v));
                    }
                    sum = HorizontalSum128(acc);
                }
            }
        }

        for (; i < vector.Length; i++)
            sum += vector[i] * vector[i];

        return MathF.Sqrt(sum);
    }

    [SkipLocalsInit]
    private static float DotProductAvx2(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int n)
    {
        float sum;
        unsafe
        {
            fixed (float* pA = a, pB = b)
            {
                var acc = Vector256<float>.Zero;
                var i = 0;
                for (; i + 8 <= n; i += 8)
                {
                    var va = Avx.LoadVector256(pA + i);
                    var vb = Avx.LoadVector256(pB + i);
                    acc = Avx.Add(acc, Avx.Multiply(va, vb));
                }
                sum = HorizontalSum256(acc);

                for (; i < n; i++)
                    sum += pA[i] * pB[i];
            }
        }
        return sum;
    }

    [SkipLocalsInit]
    private static float DotProductSse2(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int n)
    {
        float sum;
        unsafe
        {
            fixed (float* pA = a, pB = b)
            {
                var acc = Vector128<float>.Zero;
                var i = 0;
                for (; i + 4 <= n; i += 4)
                {
                    var va = Sse.LoadVector128(pA + i);
                    var vb = Sse.LoadVector128(pB + i);
                    acc = Sse.Add(acc, Sse.Multiply(va, vb));
                }
                sum = HorizontalSum128(acc);

                for (; i < n; i++)
                    sum += pA[i] * pB[i];
            }
        }
        return sum;
    }

    private static float DotProductScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int n)
    {
        var sum = 0f;
        for (var i = 0; i < n; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static void DotAndNormsAvx2(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int n,
        out float dot, out float normA, out float normB)
    {
        unsafe
        {
            fixed (float* pA = a, pB = b)
            {
                var accDot = Vector256<float>.Zero;
                var accNA = Vector256<float>.Zero;
                var accNB = Vector256<float>.Zero;
                var i = 0;
                for (; i + 8 <= n; i += 8)
                {
                    var va = Avx.LoadVector256(pA + i);
                    var vb = Avx.LoadVector256(pB + i);
                    accDot = Avx.Add(accDot, Avx.Multiply(va, vb));
                    accNA = Avx.Add(accNA, Avx.Multiply(va, va));
                    accNB = Avx.Add(accNB, Avx.Multiply(vb, vb));
                }

                dot = HorizontalSum256(accDot);
                normA = HorizontalSum256(accNA);
                normB = HorizontalSum256(accNB);

                for (; i < n; i++)
                {
                    dot += pA[i] * pB[i];
                    normA += pA[i] * pA[i];
                    normB += pB[i] * pB[i];
                }
            }
        }
    }

    private static void DotAndNormsSse2(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int n,
        out float dot, out float normA, out float normB)
    {
        unsafe
        {
            fixed (float* pA = a, pB = b)
            {
                var accDot = Vector128<float>.Zero;
                var accNA = Vector128<float>.Zero;
                var accNB = Vector128<float>.Zero;
                var i = 0;
                for (; i + 4 <= n; i += 4)
                {
                    var va = Sse.LoadVector128(pA + i);
                    var vb = Sse.LoadVector128(pB + i);
                    accDot = Sse.Add(accDot, Sse.Multiply(va, vb));
                    accNA = Sse.Add(accNA, Sse.Multiply(va, va));
                    accNB = Sse.Add(accNB, Sse.Multiply(vb, vb));
                }

                dot = HorizontalSum128(accDot);
                normA = HorizontalSum128(accNA);
                normB = HorizontalSum128(accNB);

                for (; i < n; i++)
                {
                    dot += pA[i] * pB[i];
                    normA += pA[i] * pA[i];
                    normB += pB[i] * pB[i];
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float HorizontalSum256(Vector256<float> v)
    {
        var high = Avx.ExtractVector128(v, 1);
        var low = v.GetLower();
        return HorizontalSum128(Sse.Add(high, low));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float HorizontalSum128(Vector128<float> v)
    {
        var shuf = Sse.MoveHighToLow(v, v);
        var sums = Sse.Add(v, shuf);
        sums = Sse.AddScalar(sums, Sse.Shuffle(sums, sums, 0x55));
        return sums.ToScalar();
    }
}
