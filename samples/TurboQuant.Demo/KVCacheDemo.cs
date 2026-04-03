using System.Diagnostics;
using Spectre.Console;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Simd;
using static TurboQuant.Demo.SyntheticData;

namespace TurboQuant.Demo;

internal static class KVCacheDemo
{
    internal static void Run()
    {
        const int headDim = 128;
        const int numHeads = 8;
        const int windowSize = 64;
        const int seqLen = 512;

        Output.Header("KV CACHE COMPRESSION");

        // Create per-head caches
        var heads = new (TurboQuant.Cache.KVCache Cache, TurboQuant.Cache.ResidualWindow Window)[numHeads];
        for (var h = 0; h < numHeads; h++)
        {
            (heads[h].Cache, heads[h].Window) = TurboQuantBuilder
                .Create(headDim)
                .WithKeyBits(4).WithValueBits(2)
                .WithResidualWindow(windowSize)
                .WithSeed(h * 100)
                .BuildKVCache();
        }

        // Fill
        var rng = new Random(42);
        var origKeys = new float[seqLen][];
        var origValues = new float[seqLen][];
        var sw = Stopwatch.StartNew();

        AnsiConsole.Status().Spinner(Spinner.Known.Dots2).Start($"Generating {seqLen} tokens...", _ =>
        {
            for (var t = 0; t < seqLen; t++)
            {
                var k = RandomUnitVector(headDim, rng);
                var v = RandomUnitVector(headDim, rng);
                origKeys[t] = k;
                origValues[t] = v;
                foreach (var (_, w) in heads) w.Append(k, v);
            }
        });
        sw.Stop();

        // Memory
        var (c0, w0) = heads[0];
        var origPerHead = seqLen * headDim * 2 * sizeof(float);
        var compPerHead =
            c0.Count * (PackedVector.GetPackedByteCount(headDim, 4) + PackedVector.GetPackedByteCount(headDim, 2) + 2 * sizeof(float))
            + w0.WindowCount * headDim * 2 * sizeof(float);

        var grid = new Grid().AddColumns(2);
        grid.AddRow("[dim]Config[/]", $"{numHeads} heads x {headDim}d, window={windowSize}");
        grid.AddRow("[dim]Sequence[/]", $"{seqLen} tokens ({sw.ElapsedMilliseconds}ms)");
        grid.AddRow("[dim]Tokens[/]", $"{c0.Count} compressed + {w0.WindowCount} in window");
        grid.AddRow("[dim]Memory/head[/]", $"[strikethrough]{origPerHead / 1024.0:F0} KB[/] [green]{compPerHead / 1024.0:F0} KB[/] [dim]({(1 - (double)compPerHead / origPerHead) * 100:F0}% saved)[/]");
        grid.AddRow("[dim]Total ({0} heads)[/]".Replace("{0}", $"{numHeads}"),
            $"[strikethrough]{origPerHead * numHeads / 1024:N0} KB[/] [green]{compPerHead * numHeads / 1024:N0} KB[/]");
        AnsiConsole.Write(new Panel(grid).Header("[bold]Cache[/]").Border(BoxBorder.Rounded).BorderColor(Color.Grey));

        // Attention quality
        var query = RandomUnitVector(headDim, rng);
        var allKeys = w0.GetAllKeys();
        var allValues = w0.GetAllValues();

        var compOutput = WeightedSum(allValues, Softmax(
            allKeys.Select(k => DotProduct(query, k) / MathF.Sqrt(headDim)).ToArray()));
        var exactOutput = WeightedSum(origValues, Softmax(
            origKeys.Select(k => DotProduct(query, k) / MathF.Sqrt(headDim)).ToArray()));

        AnsiConsole.WriteLine();
        var table = Output.CreateTable("Metric", "Value");
        table.AddRow("Attention output cosine", FormatCosine(DotProductSimd.CosineSimilarity(exactOutput, compOutput)));
        table.AddRow("Recent token fidelity", FormatCosine(DotProductSimd.CosineSimilarity(allKeys[^1], origKeys[^1])));
        table.AddRow("Oldest token fidelity", FormatCosine(DotProductSimd.CosineSimilarity(allKeys[0], origKeys[0])));
        AnsiConsole.Write(table);
    }

    private static string FormatCosine(float v) => v >= 0.999f ? $"[green]{v:F6}[/]" : v >= 0.99f ? $"[yellow]{v:F6}[/]" : $"{v:F6}";
}
