#!/usr/bin/env python3
"""Run the fertility.py flaw-isolation ablations (Part A, Task A2).

Single entry point: loads the corpus and tokenizers once, then calls each
selected experiment's run(corpus, encoders), which prints its essential
output and hands back (payload, csv_rows) to save under partA/results/.
Each experiment then renders its own single essential figure into
partA/plots/ via mod.plot(payload) — only for the experiments selected.

Usage
-----
    python partA/run_experiments.py
    python partA/run_experiments.py --experiment split_space
    python partA/run_experiments.py --experiment split_space lowercase
    python partA/run_experiments.py --tokenizers gpt2 hf:xlm-roberta-base
    python partA/run_experiments.py --dataset flores --no-plot
    python partA/run_experiments.py --dataset massive --raw

Flags
-----
  --experiment   one or more of the experiments listed by -h, or 'all' (default: all)
  --tokenizers   one or more tokenizer specs: 'gpt2', 'hf:<repo_id>', or 'all'
                 (gpt2, xlm-roberta-base, mt5-small, muril-base-cased, sarvam-1;
                 default: gpt2)
  --dataset      corpus name: sample (default) | sample-preprocessed | flores |
                 flores-devtest | massive, or a directory path (auto-detected as
                 FLORES *.dev, *_sample.txt, or *.txt)
  --raw          read the raw, unprocessed source instead of the preprocessed corpus:
                 FLORES *.dev/*.devtest files and MASSIVE *.jsonl (test partition),
                 verbatim — requires --dataset flores | flores-devtest | massive
  --no-plot      skip generating the selected experiments' plots
"""

import argparse
import csv
import json
import logging
import os
import sys
import unicodedata
from pathlib import Path

# Allow running this file directly: python partA/run_experiments.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import (  # noqa: E402
    exp_arg_order,
    exp_denominator,
    exp_lang_audit,
    exp_lowercase,
    exp_macro_micro,
    exp_split_space,
    exp_tok_dependence,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
PART_A = Path(__file__).resolve().parent
RESULTS_DIR = PART_A / "results"
DEFAULT_SAMPLE_DIR = ROOT / "starter_kit" / "corpus_sample"
PREPROCESSED = PART_A / "corpus" / "preprocessed_corpus"

# Dataset names accepted by --dataset -> corpus directory on disk
DATASETS = {
    "sample": DEFAULT_SAMPLE_DIR,
    "sample-preprocessed": PREPROCESSED / "sample",
    "flores": PREPROCESSED / "flores" / "dev",
    "flores-devtest": PREPROCESSED / "flores" / "devtest",
    "massive": PREPROCESSED / "massive",
}

FLORES_LANGS = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu", "mal_Mlym"]
SAMPLE_LANGS = ["eng", "hin"]

# --raw sources: dataset name -> raw directory shipped by corpus/download_corpus.py
RAW_SOURCES = {
    "flores": PART_A / "corpus" / "flores" / "flores200_dataset" / "dev",
    "flores-devtest": PART_A / "corpus" / "flores" / "flores200_dataset" / "devtest",
    "massive": PART_A / "corpus" / "massive" / "1.1" / "data",
}

# short code -> (FLORES file code, MASSIVE locale) — same mapping preprocess.py uses
CORPUS_LANGS = {
    "eng": ("eng_Latn", "en-US"),
    "hin": ("hin_Deva", "hi-IN"),
    "kan": ("kan_Knda", "kn-IN"),
    "tam": ("tam_Taml", "ta-IN"),
    "tel": ("tel_Telu", "te-IN"),
    "mal": ("mal_Mlym", "ml-IN"),
}

MASSIVE_PARTITION = "test"


# ---------------------------------------------------------------------------
# Corpus + tokenizer — read_lines/load_tokenizer copied from fertility.py;
# read_raw_lines/read_massive_raw are the verbatim readers behind --raw
# ---------------------------------------------------------------------------


def load_tokenizer(spec):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken

        enc = tiktoken.get_encoding(spec)
        return enc.encode


def read_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # normalize just in case -- some corpora are messy
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def read_raw_lines(path):
    """Read a raw FLORES file verbatim: only the line terminator is removed.

    No strip, no NFC, no whitespace collapsing — the point of --raw is to
    measure the ablations on data preprocess.py's clean() has never touched.
    """
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line:  # keep whitespace as-is; only drop truly empty lines
                lines.append(line)
    return lines


def read_massive_raw(path):
    """Read test-partition utterances from a raw MASSIVE JSONL, verbatim.

    Parsing JSON and filtering the partition are unavoidable (the raw file is
    structured); the utterance text itself is never cleaned.
    """
    utterances = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("partition") == MASSIVE_PARTITION:
                utterances.append(row["utt"])
    return utterances


def raw_paths(dataset):
    """Return (kind, paths, loader) for a --raw dataset name.

    Language keys are the short codes (eng, hin, ...) so raw runs line up
    with the preprocessed runs of the same dataset.
    """
    if dataset == "massive":
        paths = {
            short: RAW_SOURCES["massive"] / f"{locale}.jsonl"
            for short, (_, locale) in CORPUS_LANGS.items()
        }
        return "raw-massive", paths, read_massive_raw
    split = "dev" if dataset == "flores" else "devtest"
    paths = {
        short: RAW_SOURCES[dataset] / f"{code}.{split}"
        for short, (code, _) in CORPUS_LANGS.items()
    }
    return f"raw-flores-{split}", paths, read_raw_lines


def detect_dataset(dataset_dir):
    """Return (kind, paths) where kind is 'flores', 'sample', or 'custom'."""
    flores = {lang: dataset_dir / f"{lang}.dev" for lang in FLORES_LANGS}
    if all(p.exists() for p in flores.values()):
        return "flores", flores

    sample = {lang: dataset_dir / f"{lang}_sample.txt" for lang in SAMPLE_LANGS}
    if all(p.exists() for p in sample.values()):
        return "sample", sample

    found = {
        p.stem.removesuffix("_sample"): p
        for p in sorted(dataset_dir.iterdir())
        if p.suffix in {".txt", ".dev"} and not p.name.startswith(".")
    }
    if found:
        return "custom", found

    raise SystemExit(f"No recognisable corpus files found in {dataset_dir}")


# ---------------------------------------------------------------------------
# Experiments — name -> (module, one-line description for -h)
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "split_space": (exp_split_space, "split(' ') vs split() — inflated word denominator (H1)"),
    "lowercase": (exp_lowercase, "line.lower() vs case-preserved — casing effect (H2)"),
    "macro_micro": (exp_macro_micro, "macro vs micro aggregation across tpc, tps, tpb, fert (H3)"),
    "metric_choice": (exp_macro_micro, "macro vs micro aggregation across tpc, tps, tpb, fert (H3)"),
    "arg_order": (exp_arg_order, "baseline = langs[0] — ratio/label flips with order (H4)"),
    "compression": (exp_denominator, "codepoint tok/char vs byte-based CR = UTF-8 bytes/tok (H3.2)"),
    "tok_dependence": (exp_tok_dependence, "tokenizer dependence of the fertility gap (H5)"),
    "lang_audit": (exp_lang_audit, "chars/word, codepoints/token, bytes/token per language vs REPORT_v0 baseline (H7)"),
}

# --tokenizers all -> every tokenizer verified loadable in this environment
# (HF cache checked offline; ai4bharat/indic-bert excluded — gated, see plan.md)
ALL_TOKENIZERS = [
    "gpt2",
    "hf:xlm-roberta-base",
    "hf:google/mt5-small",
    "hf:google/muril-base-cased",
    "hf:sarvamai/sarvam-1",
]


def save_results(stem, payload, rows):
    """Write one experiment's JSON (plot() input) and CSV (notebook paste)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = RESULTS_DIR / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {json_path.relative_to(ROOT)}")
    print(f"  → {csv_path.relative_to(ROOT)}")


def main():
    datasets_epilog = "datasets (values for --dataset):\n" + "\n".join(
        f"  {name:<16} {path.relative_to(ROOT)}"
        for name, path in DATASETS.items()
    ) + "\n  <dir>             any directory of corpus files (layout auto-detected)"
    datasets_epilog += "\n\nraw sources (with --raw, no preprocessing):\n" + "\n".join(
        f"  {name:<16} {path.relative_to(ROOT)}"
        for name, path in RAW_SOURCES.items()
    )
    experiments_epilog = "experiments (values for --experiment):\n" + "\n".join(
        f"  {name:<14} {desc}" for name, (_, desc) in EXPERIMENTS.items()
    ) + f"\n  {'all':<14} run every experiment in order (default)"

    ap = argparse.ArgumentParser(
        description="Run the fertility.py flaw-isolation ablations (Part A, Task A2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=datasets_epilog + "\n\n" + experiments_epilog,
    )
    ap.add_argument(
        "--experiment", nargs="+", default=["all"],
        choices=[*EXPERIMENTS, "all"], metavar="EXP",
        help="which experiment(s) to run — names listed below (default: all)",
    )
    ap.add_argument(
        "--tokenizers", nargs="+", default=["gpt2"],
        metavar="TOK",
        help="tokenizer spec(s): 'gpt2', 'hf:<repo_id>', or 'all' "
             f"({', '.join(t.split(':')[-1] for t in ALL_TOKENIZERS)}; default: gpt2)",
    )
    ap.add_argument(
        "--dataset", default="sample", metavar="NAME",
        help="corpus to run on: 'sample' (10-line starter-kit toys, default), "
             "'sample-preprocessed' (same 10 lines, whitespace-collapsed), "
             "'flores' (preprocessed FLORES dev, 997 x 6), 'flores-devtest' "
             "(held-out 1012 x 6), 'massive' (preprocessed MASSIVE test, "
             "2974 x 6), or a directory path — see datasets below",
    )
    ap.add_argument("--raw", action="store_true",
                    help="read the raw, unprocessed dataset instead of the preprocessed "
                         "corpus: FLORES *.dev/*.devtest files, MASSIVE *.jsonl test "
                         "partition — requires --dataset flores | flores-devtest | massive")
    ap.add_argument("--no-plot", action="store_true", help="skip generating the selected experiments' plots")
    args = ap.parse_args()

    # --tokenizers all expands to the verified-loadable set (dedup, keep order)
    if "all" in args.tokenizers:
        args.tokenizers = list(dict.fromkeys(
            t for spec in args.tokenizers for t in (ALL_TOKENIZERS if spec == "all" else [spec])
        ))

    if args.raw:
        if args.dataset not in RAW_SOURCES:
            ap.error(
                f"--raw requires --dataset one of ({', '.join(RAW_SOURCES)}); "
                f"got: {args.dataset}"
            )
        kind, paths, loader = raw_paths(args.dataset)
        source_dir = RAW_SOURCES[args.dataset]
    else:
        if args.dataset in DATASETS:
            dataset_dir = DATASETS[args.dataset]
        else:
            dataset_dir = Path(args.dataset)
        dataset_dir = dataset_dir.resolve()
        if not dataset_dir.is_dir():
            ap.error(
                f"--dataset must be one of ({', '.join(DATASETS)}) or an existing "
                f"directory; got: {args.dataset}"
            )
        kind, paths = detect_dataset(dataset_dir)
        source_dir, loader = dataset_dir, read_lines

    print(f"Dataset   : {source_dir.relative_to(ROOT) if source_dir.is_relative_to(ROOT) else source_dir}")
    print(f"Kind      : {kind}  ({len(paths)} languages: {', '.join(paths)})")
    print(f"Tokenizers: {', '.join(args.tokenizers)}")

    # --- Load corpus once; every selected experiment reuses it ---
    corpus = {}
    for lang, path in paths.items():
        lines = loader(path)
        if not lines:
            raise SystemExit(f"empty corpus: {path}")
        corpus[lang] = lines
    print(f"Loaded    : {', '.join(f'{lang}={len(lines)}' for lang, lines in corpus.items())} lines")

    encoders = {spec: load_tokenizer(spec) for spec in args.tokenizers}

    names = list(dict.fromkeys(args.experiment))
    if "all" in names:
        names = list(EXPERIMENTS)

    for name in names:
        mod, _ = EXPERIMENTS[name]
        payload, rows = mod.run(corpus, encoders)
        save_results(mod.RESULT, {"experiment": name, **payload}, rows)
        if not args.no_plot:
            mod.plot(payload)

    print("\nResults written to:", RESULTS_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
