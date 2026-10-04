# =============================================================================
# display.py
# Tampilan terminal yang rapi (library `rich`) untuk hasil fact-checking.
# =============================================================================

from __future__ import annotations

import sys

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fact_checker.aggregation import verdict_explanation
from fact_checker.config import (
    VERDICT_DISPLAY,
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
)
from fact_checker.explainability import top_tokens
from fact_checker.schemas import FactCheckResult

# Pastikan karakter Unicode aman di terminal Windows (cp1252)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

console = Console(highlight=False)

VERDICT_STYLE = {
    VERDICT_SUPPORTED: ("green", "✔"),
    VERDICT_REFUTED: ("red", "✘"),
    VERDICT_NEI: ("yellow", "?"),
}
NLI_STYLE = {"ENTAILMENT": "green", "CONTRADICTION": "red", "NEUTRAL": "yellow"}


def print_banner(subtitle: str = "") -> None:
    title = Text("INDONESIAN NEWS FACT-CHECKER", style="bold cyan")
    body = Text.assemble(
        title,
        "\n",
        ("Hybrid Retrieval (BM25 + SBERT) · IndoBERT & Entailment Verification (NLI) · SHAP", "dim"),
    )
    if subtitle:
        body.append(f"\n{subtitle}", style="italic")
    console.print(Panel(body, box=box.DOUBLE, expand=False, padding=(0, 2)))


def _bar(value: float, width: int = 24, color: str = "cyan") -> Text:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return Text.assemble(("█" * filled, color), ("░" * (width - filled), "grey35"), f" {value:6.1%}")


def render_result(result: FactCheckResult, show_passages: bool = True) -> None:
    """Mencetak hasil lengkap ke terminal."""
    v = result.verdict
    color, icon = VERDICT_STYLE[v.label]

    console.print()
    console.print(Panel(Text(result.claim, style="bold"), title="Klaim", title_align="left",
                        border_style="cyan"))

    # ---- Verdict ----
    head = Text.assemble((f" {icon}  {VERDICT_DISPLAY[v.label]} ", f"bold white on {color}"),
                         f"   confidence {v.confidence:.1%}")
    if v.conflicting:
        head.append("   ⚠ bukti saling bertentangan", style="bold magenta")
    scores = Table.grid(padding=(0, 2))
    for label, score in v.label_scores.items():
        scores.add_row(Text(VERDICT_DISPLAY[label], style=VERDICT_STYLE[label][0]),
                       _bar(score, color=VERDICT_STYLE[label][0]))
    console.print(Panel(
        Group(head, Text(""), Text(v.reason), Text(verdict_explanation(v.label), style="dim"),
              Text(""), scores),
        title="Verdict", title_align="left", border_style=color,
    ))

    # ---- Evidence ----
    table = Table(box=box.SIMPLE_HEAVY, show_lines=show_passages, expand=True)
    table.add_column("#", justify="right", width=2)
    table.add_column("Sumber", width=16, overflow="fold")
    table.add_column("Evidence", ratio=1)
    table.add_column("Relevansi", justify="right", width=9)
    table.add_column("NLI", width=22)
    for i, ev in enumerate(result.evidences):
        mark = " ◀" if i == v.decisive_index else ""
        body = Text(ev.title, style="bold")
        if show_passages:
            body.append(f"\n{ev.passage}", style="default")
        nli = Text.assemble((ev.nli_label, NLI_STYLE[ev.nli_label]), f" {ev.nli_confidence:.0%}")
        nli.append(
            f"\nE {ev.nli_scores.get('ENTAILMENT', 0):.2f} · N {ev.nli_scores.get('NEUTRAL', 0):.2f}"
            f" · C {ev.nli_scores.get('CONTRADICTION', 0):.2f}", style="dim",
        )
        table.add_row(f"{i + 1}{mark}", f"{ev.source}\n{ev.date}", body, f"{ev.relevance:.0%}", nli)
    console.print(Panel(table, title=f"Top-{len(result.evidences)} Evidence", title_align="left",
                        border_style="blue"))

    # ---- Explainability ----
    if result.explanation:
        target = result.explanation_target
        max_abs = max((abs(val) for _, val in result.explanation), default=1.0) or 1.0
        highlighted = Text()
        for tok, val in result.explanation:
            strength = abs(val) / max_abs
            if strength < 0.15:
                highlighted.append(tok)
            else:
                bg = ("green" if val > 0 else "red") if strength > 0.5 else (
                    "dark_green" if val > 0 else "dark_red")
                highlighted.append(tok, style=f"on {bg}")
        words = Table.grid(padding=(0, 2))
        items = top_tokens(result.explanation, 10)
        span = max((abs(s) for _, s in items), default=1.0) or 1.0
        for word, score in items:
            c = "green" if score > 0 else "red"
            words.add_row(Text(("▲ " if score > 0 else "▼ ") + word, style=c),
                          Text("■" * max(1, int(abs(score) / span * 20)), style=c), f"{score:+.4f}")
        console.print(Panel(
            Group(Text(f"Kontribusi kata pada evidence penentu terhadap P({target})", style="dim"),
                  Text(""), highlighted, Text(""), words),
            title="Explainable AI (SHAP)", title_align="left", border_style="magenta",
        ))

    t = result.timings
    console.print(Text(
        "⏱  " + " · ".join(f"{k} {val:.2f}s" for k, val in t.items()), style="dim"))
