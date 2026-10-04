# =============================================================================
# __main__.py
# Command-line interface:  python -m fact_checker <perintah> [opsi]
#
#   check "klaim"     periksa satu klaim
#   interactive       mode tanya-jawab di terminal
#   demo              jalankan contoh klaim (benar / hoaks / netral)
#   app               jalankan web app Gradio
#   evaluate          evaluasi kuantitatif + laporan di folder reports/
#   build-index       bangun & cache indeks retrieval terlebih dahulu
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Izinkan `python fact_checker/__main__.py` selain `python -m fact_checker`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact_checker.config import Config  # noqa: E402

DEMO_CLAIMS = [
    "Presiden Jokowi memerintahkan Wakil Presiden Ma'ruf Amin meninjau lokasi kebakaran Depo Pertamina Plumpang.",
    "KPU menerima putusan PN Jakarta Pusat dan memutuskan tidak akan mengajukan banding.",
    "Timnas sepak bola Indonesia dipastikan menjuarai Piala Dunia 2030.",
]


def _build_config(args) -> Config:
    cfg = Config.from_env().with_overrides(
        data_path=Path(args.data) if args.data else None,
        max_articles=args.max_articles,
        top_passages=args.top_passages,
        aggregation=args.aggregation,
        device=args.device,
        nli_model=args.nli_model,
    )
    return cfg


def _setup_logging(verbose: bool) -> None:
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
    )
    for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def _load_checker(cfg: Config, load_nli: bool = True):
    from fact_checker.display import console
    from fact_checker.pipeline import FactChecker

    with console.status("[bold cyan]Memuat korpus, indeks retrieval, dan model NLI..."):
        checker = FactChecker(cfg, load_nli=load_nli)
    console.print(
        f"[green]✔[/] Sistem siap: {len(checker.articles):,} artikel · "
        f"NLI [bold]{checker.nli.model_name if checker.nli else '-'}[/] · "
        f"{checker.load_seconds:.1f} detik"
    )
    return checker


def _check_and_render(checker, claim: str, args) -> None:
    from fact_checker.display import console, render_result
    from fact_checker.explainability import plot_token_importance

    with console.status("[bold cyan]Memeriksa klaim..."):
        result = checker.check(claim, explain=not args.no_explain)
    render_result(result, show_passages=True)
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[dim]Hasil JSON disimpan ke {args.json}[/]")
    if getattr(args, "save_plot", None) and result.explanation:
        plot_token_importance(result.explanation, result.explanation_target, save_path=args.save_plot)
        console.print(f"[dim]Grafik SHAP disimpan ke {args.save_plot}[/]")


# -----------------------------------------------------------------------------
# Perintah
# -----------------------------------------------------------------------------
def cmd_check(args):
    checker = _load_checker(_build_config(args))
    _check_and_render(checker, " ".join(args.claim), args)


def cmd_interactive(args):
    from fact_checker.display import console, print_banner

    print_banner("Mode interaktif — ketik 'keluar' untuk berhenti")
    checker = _load_checker(_build_config(args))
    while True:
        try:
            claim = console.input("\n[bold cyan]› Masukkan klaim:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if claim.lower() in {"keluar", "exit", "quit", "q"}:
            break
        if len(claim.split()) < 3:
            console.print("[yellow]Klaim terlalu pendek, minimal 3 kata.[/]")
            continue
        _check_and_render(checker, claim, args)
    console.print("\nTerima kasih telah menggunakan Indonesian News Fact-Checker!")


def cmd_demo(args):
    from fact_checker.display import console, print_banner

    print_banner("Mode demo — 3 contoh klaim: benar, hoaks, di luar korpus")
    checker = _load_checker(_build_config(args))
    for i, claim in enumerate(DEMO_CLAIMS, 1):
        console.rule(f"[bold]Demo {i}/{len(DEMO_CLAIMS)}")
        args.save_plot = str(Path(args.out_dir) / f"shap_demo_{i}.png") if not args.no_explain else None
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        _check_and_render(checker, claim, args)


def cmd_app(args):
    from fact_checker.app import launch

    launch(_build_config(args), share=args.share, port=args.port)


def cmd_build_index(args):
    _load_checker(_build_config(args), load_nli=False)


def cmd_evaluate(args):
    from rich.table import Table

    from fact_checker.config import NLI_MODEL_BASELINE
    from fact_checker.display import console, print_banner
    from fact_checker.evaluation import (
        evaluate_end_to_end,
        evaluate_nli,
        evaluate_retrieval,
        load_benchmark,
        save_report,
    )
    from fact_checker.nli_model import NLIModel

    print_banner("Evaluasi kuantitatif")
    cfg = _build_config(args)
    cases = load_benchmark(args.benchmark) if args.benchmark else load_benchmark()
    console.print(f"Benchmark: {len(cases)} kasus")
    results = {"config": {k: str(v) for k, v in vars(cfg).items()}}

    checker = _load_checker(cfg)
    nli_models = {"mdeberta": checker.nli}
    if args.compare_baseline:
        nli_models["baseline"] = NLIModel(NLI_MODEL_BASELINE, device=cfg.device)

    results["nli"] = {}
    for key, model in nli_models.items():
        with console.status(f"Evaluasi NLI: {model.model_name}"):
            results["nli"][key] = evaluate_nli(model, cases)
        m = results["nli"][key]
        console.print(f"[bold]NLI {key}[/] acc={m['accuracy']:.1%} macro-F1={m['macro_f1']:.3f}")
        console.print(m["report_text"])

    if not args.skip_retrieval:
        with console.status("Evaluasi retrieval (BM25 / dense / hybrid)"):
            results["retrieval"] = evaluate_retrieval(checker.retriever, cases)
        table = Table(title="Retrieval artikel emas")
        cols = list(next(iter(results["retrieval"].values())).keys())
        table.add_column("metode")
        for c in cols:
            table.add_column(c, justify="right")
        for method, m in results["retrieval"].items():
            table.add_row(method, *[f"{m[c]:.3f}" for c in cols])
        console.print(table)

    if not args.skip_e2e:
        with console.status("Evaluasi end-to-end (klaim → verdict)"):
            results["end_to_end"] = evaluate_end_to_end(checker, cases)
        for s, m in results["end_to_end"]["strategies"].items():
            console.print(f"[bold]End-to-end ({s})[/] acc={m['accuracy']:.1%} macro-F1={m['macro_f1']:.3f}")
        console.print(results["end_to_end"]["strategies"]["max"]["report_text"])

    paths = save_report(results, cfg.reports_dir)
    for kind, p in paths.items():
        console.print(f"[green]✔[/] {kind}: {p}")


# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", help="path data.csv (default: ./data.csv)")
    common.add_argument("--max-articles", type=int, help="batasi jumlah artikel (default: semua)")
    common.add_argument("--top-passages", type=int, help="jumlah evidence yang diverifikasi")
    common.add_argument("--aggregation", choices=["max", "weighted", "mean"])
    common.add_argument("--nli-model", help="nama model NLI Hugging Face")
    common.add_argument("--device", help="cpu / cuda / mps")
    common.add_argument("-v", "--verbose", action="store_true", help="tampilkan log detail")

    parser = argparse.ArgumentParser(
        prog="python -m fact_checker",
        description="Indonesian News Fact-Checker — Hybrid Retrieval + Multilingual NLI + SHAP",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", parents=[common], help="periksa satu klaim")
    p.add_argument("claim", nargs="+")
    p.add_argument("--no-explain", action="store_true", help="lewati SHAP (lebih cepat)")
    p.add_argument("--json", help="simpan hasil ke file JSON")
    p.add_argument("--save-plot", help="simpan grafik SHAP ke PNG")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("interactive", parents=[common], help="mode interaktif")
    p.add_argument("--no-explain", action="store_true")
    p.set_defaults(func=cmd_interactive)

    p = sub.add_parser("demo", parents=[common], help="jalankan contoh klaim")
    p.add_argument("--no-explain", action="store_true")
    p.add_argument("--out-dir", default="reports", help="folder grafik SHAP")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("app", parents=[common], help="web app Gradio")
    p.add_argument("--share", action="store_true", help="buat link publik sementara")
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("evaluate", parents=[common], help="evaluasi pada benchmark")
    p.add_argument("--benchmark", help="path benchmark.jsonl")
    p.add_argument("--compare-baseline", action="store_true",
                   help="bandingkan dengan model NLI lama (English-only)")
    p.add_argument("--skip-retrieval", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("build-index", parents=[common], help="bangun cache indeks retrieval")
    p.set_defaults(func=cmd_build_index)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
