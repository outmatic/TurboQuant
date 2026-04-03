using TurboQuant.Core.Packing;
using Xunit;

namespace TurboQuant.Tests.Core;

public class BitPackerTests
{
    [Theory]
    [InlineData(2, 4)]   // 2-bit, 4 values per byte
    [InlineData(3, 8)]   // 3-bit, 8 values per 3 bytes
    [InlineData(4, 2)]   // 4-bit, 2 values per byte
    public void RoundTrip_AllBitWidths(int bits, int valuesPerUnit)
    {
        int levels = 1 << bits;
        int count = valuesPerUnit * 10; // multiple full groups

        var rng = new Random(42);
        var indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = rng.Next(levels);

        int packedBytes = PackedVector.GetPackedByteCount(count, bits);
        var packed = new byte[packedBytes];
        BitPacker.Pack(indices, packed, bits);

        var unpacked = new int[count];
        BitPacker.Unpack(packed, unpacked, count, bits);

        for (int i = 0; i < count; i++)
            Assert.Equal(indices[i], unpacked[i]);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void RoundTrip_NonAlignedCount(int bits)
    {
        int levels = 1 << bits;
        int count = 13; // non-aligned

        var rng = new Random(99);
        var indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = rng.Next(levels);

        int packedBytes = PackedVector.GetPackedByteCount(count, bits);
        var packed = new byte[packedBytes];
        BitPacker.Pack(indices, packed, bits);

        var unpacked = new int[count];
        BitPacker.Unpack(packed, unpacked, count, bits);

        for (int i = 0; i < count; i++)
            Assert.Equal(indices[i], unpacked[i]);
    }

    [Fact]
    public void CompressionRatio_4Bit()
    {
        // 4-bit: 2 indices per byte => 2x compression vs int32
        int count = 100;
        int packedBytes = PackedVector.GetPackedByteCount(count, 4);
        Assert.Equal(50, packedBytes);
    }

    [Fact]
    public void CompressionRatio_2Bit()
    {
        // 2-bit: 4 indices per byte => 4x compression vs int32
        int count = 100;
        int packedBytes = PackedVector.GetPackedByteCount(count, 2);
        Assert.Equal(25, packedBytes);
    }

    [Fact]
    public void AllZeroIndices_RoundTrip()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            var indices = new int[64];
            int packedBytes = PackedVector.GetPackedByteCount(64, bits);
            var packed = new byte[packedBytes];
            BitPacker.Pack(indices, packed, bits);

            var unpacked = new int[64];
            BitPacker.Unpack(packed, unpacked, 64, bits);

            Assert.All(unpacked, idx => Assert.Equal(0, idx));
        }
    }

    [Fact]
    public void MaxIndices_RoundTrip()
    {
        foreach (int bits in new[] { 2, 3, 4 })
        {
            int maxVal = (1 << bits) - 1;
            var indices = new int[64];
            Array.Fill(indices, maxVal);

            int packedBytes = PackedVector.GetPackedByteCount(64, bits);
            var packed = new byte[packedBytes];
            BitPacker.Pack(indices, packed, bits);

            var unpacked = new int[64];
            BitPacker.Unpack(packed, unpacked, 64, bits);

            Assert.All(unpacked, idx => Assert.Equal(maxVal, idx));
        }
    }
}
