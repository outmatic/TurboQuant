using System.Threading;

namespace TurboQuant.Diagnostics;

/// <summary>
/// Thread-safe runtime statistics for quantization operations.
/// </summary>
public sealed class CompressionStats
{
    private long _totalVectorsProcessed;
    private long _totalQuantizeTicksAccum;
    private long _dmseAccumBits; // double stored as long bits for atomic operations
    private readonly int _bits;
    private readonly int _dim;

    /// <summary>Total number of vectors quantized since last reset.</summary>
    public long TotalVectorsProcessed => Interlocked.Read(ref _totalVectorsProcessed);

    /// <summary>
    /// Average D_mse = E[||x - x̃||²] for unit vectors, as defined in the paper.
    /// </summary>
    public double AverageMSE
    {
        get
        {
            var count = TotalVectorsProcessed;
            return count > 0 ? BitConverter.Int64BitsToDouble(Interlocked.Read(ref _dmseAccumBits)) / count : 0;
        }
    }

    /// <summary>Average quantization time in milliseconds.</summary>
    public double AverageQuantizeMs
    {
        get
        {
            var count = TotalVectorsProcessed;
            return count > 0 ? (double)Interlocked.Read(ref _totalQuantizeTicksAccum) / count / TimeSpan.TicksPerMillisecond : 0;
        }
    }

    /// <summary>Compression ratio (original_bits / compressed_bits).</summary>
    public float CompressionRatio => _dim > 0 ? (32f * _dim) / (_bits * _dim + 32f) : 0f;

    internal CompressionStats(int bits, int dim)
    {
        _bits = bits;
        _dim = dim;
    }

    internal void RecordQuantize(long elapsedTicks, double dmse)
    {
        Interlocked.Increment(ref _totalVectorsProcessed);
        Interlocked.Add(ref _totalQuantizeTicksAccum, elapsedTicks);
        AtomicAddDouble(ref _dmseAccumBits, dmse);
    }

    /// <summary>Resets all counters to zero.</summary>
    public void Reset()
    {
        Interlocked.Exchange(ref _totalVectorsProcessed, 0);
        Interlocked.Exchange(ref _totalQuantizeTicksAccum, 0);
        Interlocked.Exchange(ref _dmseAccumBits, 0);
    }

    private static void AtomicAddDouble(ref long storage, double value)
    {
        var current = Interlocked.Read(ref storage);
        while (true)
        {
            var newVal = BitConverter.DoubleToInt64Bits(BitConverter.Int64BitsToDouble(current) + value);
            var prev = Interlocked.CompareExchange(ref storage, newVal, current);
            if (prev == current) break;
            current = prev;
        }
    }
}
