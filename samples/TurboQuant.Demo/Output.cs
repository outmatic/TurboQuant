using Spectre.Console;

namespace TurboQuant.Demo;

internal static class Output
{
    internal static void Banner()
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new FigletText("TurboQuant").Color(Color.DodgerBlue1));
        AnsiConsole.MarkupLine("[dim]Near-optimal vector quantization for .NET[/]");
        AnsiConsole.MarkupLine("[dim]Paper: Zandieh et al., ICLR 2026[/]");
        AnsiConsole.WriteLine();
    }

    internal static void Footer()
    {
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[green]Done.[/]");
        AnsiConsole.WriteLine();
    }

    internal static void Header(string title)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule($"[bold dodgerblue1]{title}[/]").LeftJustified());
        AnsiConsole.WriteLine();
    }

    internal static void KV(string key, string value)
        => AnsiConsole.MarkupLine($"  [dim]{key}[/]  {value}");

    internal static void KV(string key, double value, string fmt = "F4")
        => KV(key, value.ToString(fmt));

    internal static void Blank() => AnsiConsole.WriteLine();

    internal static void Line(string text) => AnsiConsole.MarkupLine($"  {text}");

    internal static void Success(string text) => AnsiConsole.MarkupLine($"  [green]{text}[/]");

    internal static void Fail(string text) => AnsiConsole.MarkupLine($"  [red]{text}[/]");

    internal static Table CreateTable(params string[] columns)
    {
        var table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
        foreach (var col in columns)
            table.AddColumn(new TableColumn(col));
        return table;
    }
}
