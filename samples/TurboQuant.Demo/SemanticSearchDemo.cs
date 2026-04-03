using System.Diagnostics;
using Spectre.Console;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Simd;
using static TurboQuant.Demo.SyntheticData;

namespace TurboQuant.Demo;

internal static class SemanticSearchDemo
{
    private static readonly string[] Topics =
        ["machine learning", "cooking recipes", "space exploration", "music theory", "quantum physics"];

    internal static void Run()
    {
        const int dim = 384;
        const int bits = 4;
        const int numDocs = 5_000;

        Output.Header("SEMANTIC SEARCH");

        var quantizer = TurboQuantBuilder.Create(dim).WithBits(bits).WithSeed(42).BuildMSE();
        var rng = new Random(42);

        // Index with progress
        var docs = new (string Topic, PackedVector Packed, float[] Raw)[numDocs];
        var sw = Stopwatch.StartNew();

        AnsiConsole.Status().Spinner(Spinner.Known.Dots2).Start($"Indexing {numDocs:N0} documents...", _ =>
        {
            for (var i = 0; i < numDocs; i++)
            {
                var emb = TopicEmbedding(dim, i % Topics.Length, rng);
                docs[i] = (Topics[i % Topics.Length], quantizer.Quantize(emb), emb);
            }
        });
        sw.Stop();

        var originalKB = (long)numDocs * dim * sizeof(float) / 1024.0;
        var compressedKB = docs.Sum(d => (long)d.Packed.Data.Length + sizeof(float)) / 1024.0;

        // Stats grid
        var grid = new Grid().AddColumns(2);
        grid.AddRow("[dim]Documents[/]", $"{numDocs:N0}");
        grid.AddRow("[dim]Dimensions[/]", $"{dim} ({bits}-bit)");
        grid.AddRow("[dim]Index time[/]", $"{sw.ElapsedMilliseconds}ms ({sw.ElapsedMilliseconds * 1000.0 / numDocs:F0}us/doc)");
        grid.AddRow("[dim]Memory[/]", $"[strikethrough]{originalKB:F0} KB[/] [green]{compressedKB:F0} KB[/] [dim]({(1 - compressedKB / originalKB) * 100:F0}% saved)[/]");
        AnsiConsole.Write(new Panel(grid).Header("[bold]Index[/]").Border(BoxBorder.Rounded).BorderColor(Color.Grey));

        // Search
        var query = TopicEmbedding(dim, topicIdx: 0, new Random(999));
        var packedQuery = quantizer.Quantize(query);

        sw.Restart();
        var results = docs
            .Select((d, i) => (d.Topic, Index: i, Score: quantizer.ApproxSimilarity(packedQuery, d.Packed)))
            .OrderByDescending(r => r.Score)
            .Take(10)
            .ToList();
        sw.Stop();

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"  Query: [bold]\"machine learning\"[/] [dim](top 10 in {sw.ElapsedMilliseconds}ms)[/]");
        AnsiConsole.WriteLine();

        var table = Output.CreateTable("#", "Doc", "Topic", "Score");
        foreach (var (r, i) in results.Select((r, i) => (r, i)))
        {
            var topicColor = r.Topic == "machine learning" ? "green" : "red";
            table.AddRow($"{i + 1}", $"{r.Index}", $"[{topicColor}]{r.Topic}[/]", $"{r.Score:F4}");
        }
        AnsiConsole.Write(table);

        var precision = results.Count(r => r.Topic == "machine learning");
        AnsiConsole.MarkupLine($"\n  Precision@10: [bold green]{precision}/10[/]");

        // Quality
        var stats = quantizer.GetStats();
        AnsiConsole.WriteLine();
        Output.KV("D_mse", stats.AverageMSE, "F6");
        Output.KV("Compression", $"{stats.CompressionRatio:F1}x");

        var exact = DotProductSimd.CosineSimilarity(query, docs[0].Raw);
        var approx = quantizer.ApproxSimilarity(packedQuery, docs[0].Packed);
        Output.KV("Cosine (exact vs approx)", $"{exact:F6} vs {approx:F6}");
    }
}
