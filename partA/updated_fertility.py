#!/usr/bin/env python3
"""Corrected tokenizer benchmark: four denominators, macro + micro, explicit baseline.

Baseline: starter_kit/fertility.py (v0).
Applies the NOTEBOOK fixes:
- whitespace collapsing (E1),
- casing preserved (E2),
- micro reported next to macro (E3),
- grapheme/byte/segment denominators (E4),
- explicit --baseline(E5),
- repeated --tokenizer (E6).

Usage:
    python partA/updated_fertility.py
    python partA/updated_fertility.py --corpus flores
    python partA/updated_fertility.py --corpus flores --tokenizer gpt2 \
        --tokenizer hf:sarvamai/sarvam-1 --baseline eng
"""

import argparse
import logging
import os
import unicodedata
from pathlib import Path

import regex as re

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]

# corpus name -> (source directory, language -> file name)
CORPUS_SOURCES = {
    "sample": (
        ROOT / "starter_kit" / "corpus_sample",
        {"eng": "eng_sample.txt", "hin": "hin_sample.txt"},
    ),
    "flores": (
        ROOT / "partA" / "corpus" / "flores" / "flores200_dataset" / "dev",
        {
            "eng": "eng_Latn.dev",
            "hin": "hin_Deva.dev",
            "kan": "kan_Knda.dev",
            "tam": "tam_Taml.dev",
            "tel": "tel_Telu.dev",
            "mal": "mal_Mlym.dev",
        },
    ),
}

DEFAULT_TOKENIZERS = ["gpt2", "hf:google/muril-base-cased"]
DEFAULT_BASELINE = "eng"

# (metric key, column label, per-line unit, decimal places):
# micro divides pooled tokens by the pooled unit total,
# macro averages the per-line ratios.
DENOMINATORS = [
    ("tokens_per_word", "tok/word", "words", 4),
    ("tokens_per_grapheme", "tok/graph", "graphemes", 4),
    ("tokens_per_byte", "tok/byte", "bytes", 4),
    ("tokens_per_segment", "tok/seg", "segments", 2),
    ("tokens_per_codepoint", "tok/cp", "codepoints", 4),
]

GRAPHEME_PATTERN = re.compile(r"\X")


def preprocess_line(raw_line):
    """NFC-normalize, trim edges, collapse whitespace runs; casing untouched."""
    trimmed = raw_line.strip()
    normalized = unicodedata.normalize("NFC", trimmed)
    return re.sub(r"\s+", " ", normalized)


def read_corpus(corpus_name):
    """Load one named corpus as {language: [preprocessed, non-empty segments]}."""
    source_dir, file_names = CORPUS_SOURCES[corpus_name]

    corpus = {}
    for language, file_name in file_names.items():
        file_path = source_dir / file_name
        if not file_path.exists():
            raise SystemExit(f"corpus file missing: {file_path}")

        segments = []
        with file_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                segment = preprocess_line(raw_line)
                if segment:
                    segments.append(segment)

        if not segments:
            raise SystemExit(f"empty corpus: {file_path}")
        corpus[language] = segments

    return corpus


def validate_parallel_alignment(corpus):
    """TPS needs equal non-empty segment counts across every language."""
    segment_counts = {language: len(segments) for language, segments in corpus.items()}

    if len(set(segment_counts.values())) != 1:
        counts = ", ".join(f"{lang}={count}" for lang, count in segment_counts.items())
        raise SystemExit(
            f"error: corpora are not parallel ({counts}); "
            "tokens per parallel sentence requires equal segment counts"
        )

    return segment_counts


def load_tokenizer(spec):
    """Return encode(text) -> token ids; special tokens excluded everywhere."""
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec[3:])
        return lambda text: tokenizer.encode(text, add_special_tokens=False)

    import tiktoken

    return tiktoken.get_encoding(spec).encode


def measure_language(lines, encode):
    """Pool totals and both aggregations for every denominator over one language."""
    totals = {
        "segments": 0,
        "tokens": 0,
        "words": 0,
        "graphemes": 0,
        "bytes": 0,
        "codepoints": 0,
    }
    macro_sums = {key: 0.0 for key, _, _, _ in DENOMINATORS}

    for line in lines:
        token_count = len(encode(line))
        line_units = {
            "segments": 1,
            "tokens": token_count,
            "words": len(line.split()),
            "graphemes": len(GRAPHEME_PATTERN.findall(line)),
            "bytes": len(line.encode("utf-8")),
            "codepoints": len(line),
        }

        for unit_name, unit_count in line_units.items():
            totals[unit_name] += unit_count
        for key, _, unit_name, _ in DENOMINATORS:
            macro_sums[key] += token_count / line_units[unit_name]

    segment_count = totals["segments"]

    return {
        "segment_count": segment_count,
        "total_tokens": totals["tokens"],
        "macro": {key: macro_sums[key] / segment_count for key, _, _, _ in DENOMINATORS},
        "micro": {key: totals["tokens"] / totals[unit] for key, _, unit, _ in DENOMINATORS},
    }


def print_metrics_table(tokenizer_spec, title, per_language, aggregation):
    """Print one denominator table (macro or micro) for a single tokenizer."""
    print(f"\n[{tokenizer_spec}] {title}")
    header = (
        f"{'lang':<8}{'segments':>9}{'tokens':>10}"
        + "".join(f"{label:>12}" for _, label, _, _ in DENOMINATORS)
    )
    print(header)
    print("-" * len(header))

    for language, measured in per_language.items():
        values = measured[aggregation]
        cells = "".join(
            f"{values[key]:>12.{precision}f}" for key, _, _, precision in DENOMINATORS
        )
        print(
            f"{language:<8}{measured['segment_count']:>9}{measured['total_tokens']:>10}{cells}"
        )


def print_ratio_table(per_language, baseline_language):
    """Print language-to-baseline ratios for every denominator, both aggregations."""
    baseline = per_language[baseline_language]

    print(f"\nratios vs baseline '{baseline_language}' (micro; macro in parentheses)")
    header = f"{'lang':<8}" + "".join(f"{label:>17}" for _, label, _, _ in DENOMINATORS)
    print(header)
    print("-" * len(header))

    for language, measured in per_language.items():
        cells = []
        for key, _, _, _ in DENOMINATORS:
            micro_ratio = measured["micro"][key] / baseline["micro"][key]
            macro_ratio = measured["macro"][key] / baseline["macro"][key]
            cells.append(f"{micro_ratio:.2f} ({macro_ratio:.2f})")
        print(f"{language:<8}" + "".join(f"{cell:>17}" for cell in cells))


def print_interpretation():
    """State which number drives routing-and-cost decisions, and the caveat."""
    print("\ninterpretation")
    print("  primary number: micro tokens per parallel sentence (TPS).")
    print("  A routing-and-cost decision asks how many model tokens it takes to")
    print("  process equivalent content. A parallel sentence is the closest denominator")
    print("  in this experiment to holding that content constant, so micro TPS is the")
    print("  number to compare across languages. Fertility and the character-based")
    print("  metrics describe tokenizer behaviour, but their units (words, bytes,")
    print("  codepoints, graphemes) shift between scripts, so they do not measure the")
    print("  token budget for equivalent meaning.")
    print("  Caveat: TPS is the right number for this cross-language comparison, not a")
    print("  universal production-cost estimate — real routing should eventually use")
    print("  token counts from actual production requests.")
    print("  Micro is the corpus-level figure; macro shows average sentence-level behaviour.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Corrected tokenizer benchmark: four denominators, macro + micro, explicit baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python partA/updated_fertility.py\n"
            "  python partA/updated_fertility.py --corpus flores\n"
            "  python partA/updated_fertility.py --corpus flores --tokenizer gpt2 "
            "--tokenizer hf:sarvamai/sarvam-1 --baseline hin\n"
        ),
    )
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_SOURCES),
        default="sample",
        help="corpus to measure: 'sample' (10-line starter-kit, default) or "
        "'flores' (997 parallel FLORES-200 dev sentences, 6 languages)",
    )
    parser.add_argument(
        "--tokenizer",
        action="append",
        dest="tokenizers",
        metavar="SPEC",
        help="tokenizer spec, repeatable: 'gpt2' or 'hf:<repo_id>'; "
        "default: gpt2 + hf:google/muril-base-cased",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        metavar="LANG",
        help="language others are compared against (default: eng; "
        "independent of corpus or argument order)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    tokenizer_specs = list(dict.fromkeys(args.tokenizers or DEFAULT_TOKENIZERS))

    corpus = read_corpus(args.corpus)
    segment_counts = validate_parallel_alignment(corpus)

    if args.baseline not in corpus:
        available = ", ".join(corpus)
        raise SystemExit(
            f"error: --baseline {args.baseline!r} not in corpus languages: {available}"
        )

    print(f"corpus        : {args.corpus}")
    counts_text = ", ".join(f"{lang}={count}" for lang, count in segment_counts.items())
    print(f"segments      : {counts_text}  (parallel: yes)")
    print(
        "preprocessing : NFC + whitespace collapse, casing preserved; "
        "identical text for every tokenizer"
    )
    print(f"tokenizers    : {', '.join(tokenizer_specs)}")
    print(f"baseline      : {args.baseline}")
    print("note          : tok/cp is a within-language diagnostic; graphemes are the per-character denominator")

    for spec in tokenizer_specs:
        encode = load_tokenizer(spec)
        per_language = {
            language: measure_language(lines, encode)
            for language, lines in corpus.items()
        }

        print_metrics_table(spec, "macro (mean of per-line ratios)", per_language, "macro")
        print_metrics_table(spec, "micro (pooled / pooled; primary)", per_language, "micro")
        print_ratio_table(per_language, args.baseline)

    print_interpretation()


if __name__ == "__main__":
    main()
