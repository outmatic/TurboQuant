using System.Runtime.CompilerServices;

namespace TurboQuant.Core.Packing;

/// <summary>
/// Packs and unpacks quantization indices into compact byte arrays.
/// Supports 2-bit, 3-bit, and 4-bit packing with zero allocation via Span.
/// </summary>
public static class BitPacker
{
    /// <summary>
    /// Packs quantization indices into bytes.
    /// </summary>
    /// <param name="indices">Source indices (each in range [0, 2^bits - 1]).</param>
    /// <param name="output">Destination byte span. Must be at least GetPackedByteCount(indices.Length, bits) bytes.</param>
    /// <param name="bits">Bit width (2, 3, or 4).</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Pack(ReadOnlySpan<int> indices, Span<byte> output, int bits)
    {
        switch (bits)
        {
            case 2: Pack2Bit(indices, output); break;
            case 3: Pack3Bit(indices, output); break;
            case 4: Pack4Bit(indices, output); break;
            default: throw new ArgumentOutOfRangeException(nameof(bits), "Supported: 2, 3, 4.");
        }
    }

    /// <summary>
    /// Unpacks bytes back to quantization indices.
    /// </summary>
    /// <param name="packed">Source packed bytes.</param>
    /// <param name="output">Destination index span.</param>
    /// <param name="count">Number of indices to unpack.</param>
    /// <param name="bits">Bit width (2, 3, or 4).</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Unpack(ReadOnlySpan<byte> packed, Span<int> output, int count, int bits)
    {
        switch (bits)
        {
            case 2: Unpack2Bit(packed, output, count); break;
            case 3: Unpack3Bit(packed, output, count); break;
            case 4: Unpack4Bit(packed, output, count); break;
            default: throw new ArgumentOutOfRangeException(nameof(bits), "Supported: 2, 3, 4.");
        }
    }

    // --- 4-bit: 2 indices per byte, little-endian ---

    private static void Pack4Bit(ReadOnlySpan<int> indices, Span<byte> output)
    {
        var pairs = indices.Length / 2;
        for (var i = 0; i < pairs; i++)
        {
            output[i] = (byte)((indices[i * 2] & 0xF) | ((indices[i * 2 + 1] & 0xF) << 4));
        }
        if ((indices.Length & 1) != 0)
        {
            output[pairs] = (byte)(indices[indices.Length - 1] & 0xF);
        }
    }

    private static void Unpack4Bit(ReadOnlySpan<byte> packed, Span<int> output, int count)
    {
        var pairs = count / 2;
        for (var i = 0; i < pairs; i++)
        {
            output[i * 2] = packed[i] & 0xF;
            output[i * 2 + 1] = (packed[i] >> 4) & 0xF;
        }
        if ((count & 1) != 0)
        {
            output[count - 1] = packed[pairs] & 0xF;
        }
    }

    // --- 2-bit: 4 indices per byte ---

    private static void Pack2Bit(ReadOnlySpan<int> indices, Span<byte> output)
    {
        var quads = indices.Length / 4;
        for (var i = 0; i < quads; i++)
        {
            var baseIdx = i * 4;
            output[i] = (byte)(
                (indices[baseIdx] & 0x3) |
                ((indices[baseIdx + 1] & 0x3) << 2) |
                ((indices[baseIdx + 2] & 0x3) << 4) |
                ((indices[baseIdx + 3] & 0x3) << 6));
        }
        var remaining = indices.Length % 4;
        if (remaining > 0)
        {
            var val = (byte)0;
            var baseIdx = quads * 4;
            for (var j = 0; j < remaining; j++)
                val |= (byte)((indices[baseIdx + j] & 0x3) << (j * 2));
            output[quads] = val;
        }
    }

    private static void Unpack2Bit(ReadOnlySpan<byte> packed, Span<int> output, int count)
    {
        var quads = count / 4;
        for (var i = 0; i < quads; i++)
        {
            var b = packed[i];
            var baseIdx = i * 4;
            output[baseIdx] = b & 0x3;
            output[baseIdx + 1] = (b >> 2) & 0x3;
            output[baseIdx + 2] = (b >> 4) & 0x3;
            output[baseIdx + 3] = (b >> 6) & 0x3;
        }
        var remaining = count % 4;
        if (remaining > 0)
        {
            var b = packed[quads];
            var baseIdx = quads * 4;
            for (var j = 0; j < remaining; j++)
                output[baseIdx + j] = (b >> (j * 2)) & 0x3;
        }
    }

    // --- 3-bit: 8 indices per 3 bytes ---
    // Layout: indices i0..i7 packed as 24 bits (3 bytes) in little-endian:
    // byte0 = i0[2:0] | i1[2:0] << 3 | i2[1:0] << 6
    // byte1 = i2[2] | i3[2:0] << 1 | i4[2:0] << 4 | i5[0] << 7
    // byte2 = i5[2:1] | i6[2:0] << 2 | i7[2:0] << 5

    private static void Pack3Bit(ReadOnlySpan<int> indices, Span<byte> output)
    {
        var groups = indices.Length / 8;
        for (var g = 0; g < groups; g++)
        {
            var si = g * 8;
            var di = g * 3;
            var i0 = indices[si] & 0x7;
            var i1 = indices[si + 1] & 0x7;
            var i2 = indices[si + 2] & 0x7;
            var i3 = indices[si + 3] & 0x7;
            var i4 = indices[si + 4] & 0x7;
            var i5 = indices[si + 5] & 0x7;
            var i6 = indices[si + 6] & 0x7;
            var i7 = indices[si + 7] & 0x7;

            output[di]     = (byte)(i0 | (i1 << 3) | ((i2 & 0x3) << 6));
            output[di + 1] = (byte)((i2 >> 2) | (i3 << 1) | (i4 << 4) | ((i5 & 0x1) << 7));
            output[di + 2] = (byte)((i5 >> 1) | (i6 << 2) | (i7 << 5));
        }

        var remaining = indices.Length % 8;
        if (remaining > 0)
        {
            // Pack remaining using a general bit-stuffing approach
            var si = groups * 8;
            var di = groups * 3;
            var bitBuffer = 0;
            var bitsUsed = 0;
            var byteOffset = 0;
            for (var j = 0; j < remaining; j++)
            {
                bitBuffer |= (indices[si + j] & 0x7) << bitsUsed;
                bitsUsed += 3;
                while (bitsUsed >= 8)
                {
                    output[di + byteOffset] = (byte)(bitBuffer & 0xFF);
                    bitBuffer >>= 8;
                    bitsUsed -= 8;
                    byteOffset++;
                }
            }
            if (bitsUsed > 0)
                output[di + byteOffset] = (byte)(bitBuffer & 0xFF);
        }
    }

    private static void Unpack3Bit(ReadOnlySpan<byte> packed, Span<int> output, int count)
    {
        var groups = count / 8;
        for (var g = 0; g < groups; g++)
        {
            var si = g * 3;
            var di = g * 8;
            var b0 = (int)packed[si];
            var b1 = (int)packed[si + 1];
            var b2 = (int)packed[si + 2];

            output[di]     = b0 & 0x7;
            output[di + 1] = (b0 >> 3) & 0x7;
            output[di + 2] = ((b0 >> 6) | (b1 << 2)) & 0x7;
            output[di + 3] = (b1 >> 1) & 0x7;
            output[di + 4] = (b1 >> 4) & 0x7;
            output[di + 5] = ((b1 >> 7) | (b2 << 1)) & 0x7;
            output[di + 6] = (b2 >> 2) & 0x7;
            output[di + 7] = (b2 >> 5) & 0x7;
        }

        var remaining = count % 8;
        if (remaining > 0)
        {
            var si = groups * 3;
            var di = groups * 8;
            var bitBuffer = 0;
            var bitsLoaded = 0;
            var byteIdx = 0;
            for (var j = 0; j < remaining; j++)
            {
                while (bitsLoaded < 3 && si + byteIdx < packed.Length)
                {
                    bitBuffer |= packed[si + byteIdx] << bitsLoaded;
                    bitsLoaded += 8;
                    byteIdx++;
                }
                output[di + j] = bitBuffer & 0x7;
                bitBuffer >>= 3;
                bitsLoaded -= 3;
            }
        }
    }
}
