namespace TurboQuant.Demo;

internal static class SyntheticData
{
    internal static float[] RandomUnitVector(int dim, Random rng)
    {
        var v = new float[dim];
        var norm = 0f;
        for (var i = 0; i < dim; i++)
        {
            v[i] = (float)BoxMuller(rng);
            norm += v[i] * v[i];
        }
        norm = MathF.Sqrt(norm);
        for (var i = 0; i < dim; i++) v[i] /= norm;
        return v;
    }

    internal static float[] TopicEmbedding(int dim, int topicIdx, Random rng)
    {
        var topicRng = new Random(topicIdx * 1000);
        var v = new float[dim];
        for (var i = 0; i < dim; i++)
            v[i] = (float)BoxMuller(topicRng) + (float)(BoxMuller(rng) * 0.3);

        var norm = 0f;
        for (var i = 0; i < dim; i++) norm += v[i] * v[i];
        norm = MathF.Sqrt(norm);
        for (var i = 0; i < dim; i++) v[i] /= norm;
        return v;
    }

    internal static float DotProduct(float[] a, float[] b)
    {
        var sum = 0f;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
        var sum = exps.Sum();
        return exps.Select(e => e / sum).ToArray();
    }

    internal static float[] WeightedSum(float[][] vectors, float[] weights)
    {
        var dim = vectors[0].Length;
        var result = new float[dim];
        for (var i = 0; i < vectors.Length; i++)
            for (var j = 0; j < dim; j++)
                result[j] += weights[i] * vectors[i][j];
        return result;
    }

    private static double BoxMuller(Random rng)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
