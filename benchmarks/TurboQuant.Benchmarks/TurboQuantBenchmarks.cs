using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using TurboQuant;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Quantizers;
using TurboQuant.Core.Rotation;
using TurboQuant.Core.Simd;

namespace TurboQuant.Benchmarks;

[MemoryDiagnoser]
[SimpleJob]
public class TurboQuantBenchmarks
{
    private TurboQuantMSE _quantizer = null!;
    private float[] _vector = null!;
    private float[] _batchVectors = null!;
    private PackedVector _packed;
    private PackedVector _packedB;
    private PackedVector[] _batchOutput = null!;

    [Params(256, 768)]
    public int Dimension { get; set; }

    [Params(4)]
    public int Bits { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _quantizer = new TurboQuantMSE(Dimension, Bits, seed: 42);

        var rng = new Random(42);
        _vector = new float[Dimension];
        for (int i = 0; i < Dimension; i++)
            _vector[i] = (float)(rng.NextDouble() * 2 - 1);

        _packed = _quantizer.Quantize(_vector);

        var v2 = new float[Dimension];
        for (int i = 0; i < Dimension; i++)
            v2[i] = (float)(rng.NextDouble() * 2 - 1);
        _packedB = _quantizer.Quantize(v2);

        _batchVectors = new float[Dimension * 100];
        for (var i = 0; i < _batchVectors.Length; i++)
            _batchVectors[i] = (float)(rng.NextDouble() * 2 - 1);

        _batchOutput = new PackedVector[100];
    }

    [Benchmark]
    public PackedVector Quantize() => _quantizer.Quantize(_vector);

    [Benchmark]
    public float[] Dequantize() => _quantizer.Dequantize(_packed);

    [Benchmark]
    public void QuantizeBatch_100() => _quantizer.QuantizeBatch(_batchVectors, _batchOutput);

    [Benchmark]
    public float ApproxSimilarity() => _quantizer.ApproxSimilarity(_packed, _packedB);

    [Benchmark]
    public float DotProduct_SIMD() => DotProductSimd.DotProduct(_vector, _vector);

    [Benchmark]
    public float L2Norm_SIMD() => DotProductSimd.L2Norm(_vector);
}

[MemoryDiagnoser]
[SimpleJob]
public class HadamardBenchmarks
{
    private HadamardRotation _rotation = null!;
    private float[] _vector = null!;

    [Params(256, 768, 1024)]
    public int Dimension { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _rotation = new HadamardRotation(Dimension, seed: 42);

        var paddedDim = _rotation.PaddedDimension;
        _vector = new float[paddedDim];
        var rng = new Random(42);
        for (var i = 0; i < Dimension; i++)
            _vector[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    [Benchmark]
    public void HadamardTransform() => _rotation.Transform(_vector);

    [Benchmark]
    public void HadamardInverse() => _rotation.InverseTransform(_vector);

    [Benchmark]
    public void HadamardRoundTrip()
    {
        _rotation.Transform(_vector);
        _rotation.InverseTransform(_vector);
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        BenchmarkRunner.Run(new[] { typeof(TurboQuantBenchmarks), typeof(HadamardBenchmarks) }, args: args);
    }
}
