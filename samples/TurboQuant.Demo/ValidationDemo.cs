using Spectre.Console;
using TurboQuant.Diagnostics;

namespace TurboQuant.Demo;

internal static class ValidationDemo
{
    internal static void Run()
    {
        Output.Header("PAPER VALIDATION");

        var table = Output.CreateTable("Bits", "D_mse", "Theoretical", "Delta", "Status");

        foreach (var bits in new[] { 4, 3, 2 })
        {
            var r = QuantizationBenchmark.RunValidation(dim: 768, bits: bits, numVectors: 2000);
            var status = r.Passed ? "[green]PASS[/]" : "[red]FAIL[/]";
            var deltaColor = r.DeltaPercent < 1 ? "green" : r.DeltaPercent < 3 ? "yellow" : "red";
            table.AddRow(
                $"{bits}-bit",
                $"{r.EmpiricalMSE:F6}",
                $"{r.TheoreticalMSE:F6}",
                $"[{deltaColor}]{r.DeltaPercent:F2}%[/]",
                status);
        }

        AnsiConsole.Write(table);
    }
}
