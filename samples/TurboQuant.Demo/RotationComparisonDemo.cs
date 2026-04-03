using System.Diagnostics;
using Spectre.Console;
using TurboQuant.Core.Packing;
using TurboQuant.Core.Quantizers;
using TurboQuant.Core.Rotation;
using TurboQuant.Core.Simd;
using static TurboQuant.Demo.SyntheticData;

namespace TurboQuant.Demo;

internal static class RotationComparisonDemo
{
    internal static void Run()
    {
        Output.Header("ROTATION COMPARISON: RANDOM vs HADAMARD");

        var dims = new[] { 256, 768, 1024 };
        const int bits = 4;
        const int numVectors = 500;

        var table = Output.CreateTable(
            "Dim", "Rotation", "Padded", "Quantize", "Similarity", "D_mse", "Packed Size");

        foreach (var dim in dims)
        {
            var rng = new Random(999);
            var vectors = Enumerable.Range(0, numVectors)
                .Select(_ => RandomUnitVector(dim, rng))
                .ToArray();

            foreach (var useHadamard in new[] { false, true })
            {
                var label = useHadamard ? "Hadamard" : "Random";
                IRotation rotation = useHadamard
                    ? new HadamardRotation(dim, seed: 42)
                    : new RandomRotation(dim, seed: 42);
                var quantizer = new TurboQuantMSE(dim, bits, rotation);
                var paddedDim = quantizer.RotationDimension;

                // Quantize all vectors, measure time
                var packed = new PackedVector[numVectors];
                var sw = Stopwatch.StartNew();
                for (var i = 0; i < numVectors; i++)
                    packed[i] = quantizer.Quantize(vectors[i]);
                sw.Stop();
                var usPerVec = sw.Elapsed.TotalMicroseconds / numVectors;

                // Measure cosine similarity
                var totalCosine = 0.0;
                var totalDmse = 0.0;
                for (var i = 0; i < numVectors; i++)
                {
                    var restored = quantizer.Dequantize(packed[i]);
                    totalCosine += DotProductSimd.CosineSimilarity(vectors[i], restored);
                    for (var j = 0; j < dim; j++)
                    {
                        var diff = vectors[i][j] - restored[j];
                        totalDmse += diff * diff;
                    }
                }
                var avgCosine = totalCosine / numVectors;
                var avgDmse = totalDmse / numVectors;
                var packedBytes = packed[0].Data.Length;

                var padNote = paddedDim == dim ? $"{paddedDim}" : $"[yellow]{paddedDim}[/]";
                var cosColor = avgCosine >= 0.995 ? "green" : avgCosine >= 0.99 ? "yellow" : "white";
                var rotColor = useHadamard ? "cyan" : "blue";

                table.AddRow(
                    $"{dim}",
                    $"[{rotColor}]{label}[/]",
                    padNote,
                    $"{usPerVec:F0} us",
                    $"[{cosColor}]{avgCosine:F6}[/]",
                    $"{avgDmse:F6}",
                    $"{packedBytes} B");
            }

            // Add separator between dims (except last)
            if (dim != dims[^1])
                table.AddEmptyRow();
        }

        AnsiConsole.Write(table);

        // Summary
        AnsiConsole.WriteLine();
        Output.Line("[dim]Random:[/] paper-correct, any dimension, no padding overhead");
        Output.Line("[dim]Hadamard:[/] faster, but pads non-power-of-2 dims (more storage)");
        AnsiConsole.WriteLine();
        Output.Line("[yellow]Note:[/] D_mse is not directly comparable between rotations when padding is involved.");
        Output.Line("[yellow]      [/] Hadamard at dim=768 quantizes 1024 coordinates (256 are padded zeros),");
        Output.Line("[yellow]      [/] diluting the error. Cosine similarity is the reliable quality metric.");
        AnsiConsole.WriteLine();
        Output.Line("[dim]Tip:[/] use power-of-2 dimensions (512, 1024) to get Hadamard speed with zero overhead");
    }
}
