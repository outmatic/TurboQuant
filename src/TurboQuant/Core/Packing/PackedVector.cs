using System.Buffers.Binary;
using System.Runtime.CompilerServices;

namespace TurboQuant.Core.Packing;

/// <summary>
/// A quantized vector packed into a compact byte representation.
/// Data is immutable after creation. Includes serialization support.
/// </summary>
public readonly struct PackedVector
{
    /// <summary>Raw packed byte data containing quantization indices (immutable).</summary>
    public ReadOnlyMemory<byte> Data { get; }

    /// <summary>L2 norm of the original vector.</summary>
    public float Norm { get; }

    /// <summary>Original (unpadded) vector dimension.</summary>
    public int Dim { get; }

    /// <summary>Number of bits per quantized coordinate.</summary>
    public int Bits { get; }

    /// <summary>Padded/rotation dimension.</summary>
    public int PaddedDim { get; }

    internal PackedVector(byte[] data, float norm, int dim, int bits, int paddedDim)
    {
        Data = data;
        Norm = norm;
        Dim = dim;
        Bits = bits;
        PaddedDim = paddedDim;
    }

    /// <summary>Compression ratio compared to float32 representation.</summary>
    public float CompressionRatio => (Dim * 32f) / (Data.Length * 8f + 32f);

    /// <summary>
    /// Computes the expected number of packed bytes for the given dimension and bit width.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetPackedByteCount(int paddedDim, int bits) =>
        bits switch
        {
            2 => (paddedDim + 3) / 4,
            3 => (paddedDim * 3 + 7) / 8,
            4 => (paddedDim + 1) / 2,
            _ => throw new ArgumentOutOfRangeException(nameof(bits))
        };

    // ── Serialization ──────────────────────────────
    // Format: [Dim:4][Bits:4][PaddedDim:4][Norm:4][DataLength:4][Data:N]
    // Total header: 20 bytes, little-endian.

    private const int HeaderSize = 20;

    /// <summary>Total number of bytes when serialized.</summary>
    public int SerializedSize => HeaderSize + Data.Length;

    /// <summary>Serializes this PackedVector into the destination span.</summary>
    public void WriteTo(Span<byte> destination)
    {
        BinaryPrimitives.WriteInt32LittleEndian(destination, Dim);
        BinaryPrimitives.WriteInt32LittleEndian(destination[4..], Bits);
        BinaryPrimitives.WriteInt32LittleEndian(destination[8..], PaddedDim);
        BinaryPrimitives.WriteSingleLittleEndian(destination[12..], Norm);
        BinaryPrimitives.WriteInt32LittleEndian(destination[16..], Data.Length);
        Data.Span.CopyTo(destination[HeaderSize..]);
    }

    /// <summary>Serializes to a new byte array.</summary>
    public byte[] ToBytes()
    {
        var result = new byte[SerializedSize];
        WriteTo(result);
        return result;
    }

    /// <summary>Deserializes a PackedVector from a byte span.</summary>
    public static PackedVector FromBytes(ReadOnlySpan<byte> source)
    {
        var dim = BinaryPrimitives.ReadInt32LittleEndian(source);
        var bits = BinaryPrimitives.ReadInt32LittleEndian(source[4..]);
        var paddedDim = BinaryPrimitives.ReadInt32LittleEndian(source[8..]);
        var norm = BinaryPrimitives.ReadSingleLittleEndian(source[12..]);
        var dataLen = BinaryPrimitives.ReadInt32LittleEndian(source[16..]);
        var data = source.Slice(HeaderSize, dataLen).ToArray();
        return new(data, norm, dim, bits, paddedDim);
    }

    /// <summary>Number of bytes consumed by a serialized PackedVector starting at this span.</summary>
    public static int ReadSerializedSize(ReadOnlySpan<byte> source) =>
        HeaderSize + BinaryPrimitives.ReadInt32LittleEndian(source[16..]);
}
