using TurboQuant.Core.Rotation;
using Xunit;

namespace TurboQuant.Tests.Core;

public class HadamardRotationTests
{
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(768)]
    public void Transform_InverseTransform_IsIdentity(int dim)
    {
        var rotation = new HadamardRotation(dim, seed: 42);
        int paddedDim = rotation.PaddedDimension;

        var original = new float[paddedDim];
        var rng = new Random(123);
        for (int i = 0; i < dim; i++)
            original[i] = (float)(rng.NextDouble() * 2 - 1);

        var copy = original.ToArray();

        rotation.Transform(copy.AsSpan());
        rotation.InverseTransform(copy.AsSpan());

        for (int i = 0; i < dim; i++)
            Assert.InRange(copy[i] - original[i], -1e-4f, 1e-4f);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(768)]
    public void Transform_PreservesNorm(int dim)
    {
        var rotation = new HadamardRotation(dim, seed: 99);
        int paddedDim = rotation.PaddedDimension;

        var vector = new float[paddedDim];
        var rng = new Random(456);
        float normBefore = 0;
        for (int i = 0; i < dim; i++)
        {
            vector[i] = (float)(rng.NextDouble() * 2 - 1);
            normBefore += vector[i] * vector[i];
        }
        normBefore = MathF.Sqrt(normBefore);

        rotation.Transform(vector.AsSpan());

        float normAfter = 0;
        for (int i = 0; i < paddedDim; i++)
            normAfter += vector[i] * vector[i];
        normAfter = MathF.Sqrt(normAfter);

        Assert.InRange(normAfter, normBefore * 0.999f, normBefore * 1.001f);
    }

    [Fact]
    public void SameSeed_ProducesSameResults()
    {
        var rot1 = new HadamardRotation(128, seed: 42);
        var rot2 = new HadamardRotation(128, seed: 42);

        var v1 = new float[128];
        var v2 = new float[128];
        var rng = new Random(1);
        for (int i = 0; i < 128; i++)
            v1[i] = v2[i] = (float)rng.NextDouble();

        rot1.Transform(v1);
        rot2.Transform(v2);

        for (int i = 0; i < 128; i++)
            Assert.Equal(v1[i], v2[i], precision: 6);
    }

    [Fact]
    public void NonPowerOfTwo_PadsCorrectly()
    {
        var rotation = new HadamardRotation(100, seed: 42);
        Assert.Equal(128, rotation.PaddedDimension);

        rotation = new HadamardRotation(768, seed: 42);
        Assert.Equal(1024, rotation.PaddedDimension);
    }
}
