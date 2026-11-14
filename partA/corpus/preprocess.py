"""Preprocess eval corpus

Indic text has multiple byte representations for the same visible characters,
so normalize form (NFC by default, selectable NFKC) and whitespace runs while
preserving linguistic content (casing, word structure).
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

FLORES_SPLITS = ("dev", "devtest")
MASSIVE_PARTITION = "test"

LANGUAGES = [
    ("eng", "eng_Latn", "en-US"),
    ("hin", "hin_Deva", "hi-IN"),
    ("kan", "kan_Knda", "kn-IN"),
    ("tam", "tam_Taml", "ta-IN"),
    ("tel", "tel_Telu", "te-IN"),
    ("mal", "mal_Mlym", "ml-IN"),
]

BASE = Path(__file__).resolve().parent
FLORES_RAW = BASE / "flores" / "flores200_dataset"
MASSIVE_RAW = BASE / "massive" / "1.1" / "data"
SAMPLE_RAW = BASE.parent.parent / "starter_kit" / "corpus_sample"
SAMPLE_LANGS = ["eng", "hin"]
DEFAULT_OUT = BASE / "preprocessed_corpus"

class PreprocessError(Exception):
    """Raised when a raw corpus is missing or violates parallel alignment."""

def clean(text: str, form: str = "NFC") -> str:
    """Normalize to ``form``, strip edge whitespace, and collapse whitespace runs.

    ``form`` is "NFC" (default) or "NFKC"; NFKC additionally applies
    compatibility decomposition (e.g. ``km²`` -> ``km2``), which can alter
    tokenization-relevant characters.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize(form, text.strip()))


def read_lines(path: Path) -> list[str]:
    """Read a raw corpus file, failing with a hint when it is absent."""
    if not path.exists():
        raise PreprocessError(f"{path} missing; run download_corpus.py first")

    return path.read_text(encoding="utf-8").splitlines()


def load_flores(split: str) -> dict[str, list[str]]:
    """Read one raw FLORES split as lines per language."""
    return {
        short: read_lines(FLORES_RAW / split / f"{code}.{split}")
        for short, code, _ in LANGUAGES
    }


def load_massive() -> dict[str, list[str]]:
    """Read the MASSIVE test partition as utterances per locale."""
    rows: dict[str, list[str]] = {}
    for short, _, locale in LANGUAGES:
        source = MASSIVE_RAW / f"{locale}.jsonl"
        utterances: list[str] = []

        for number, raw in enumerate(read_lines(source), 1):
            if not raw.strip():
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PreprocessError(f"{source}:{number}: invalid JSON ({exc})") from exc

            if row.get("partition") == MASSIVE_PARTITION:
                utterances.append(row["utt"])

        rows[short] = utterances

    return rows


def load_sample() -> dict[str, list[str]]:
    """Read the starter-kit sample corpus (2 languages, 10 lines each)."""
    return {
        short: read_lines(SAMPLE_RAW / f"{short}_sample.txt")
        for short in SAMPLE_LANGS
    }


def preprocess(rows: dict[str, list[str]], target: Path, label: str, norm: str = "NFC") -> None:
    """Canonicalize, verify parallel alignment, and write one file per language."""
    cleaned = {
        short: [text for line in lines if (text := clean(line, norm))]
        for short, lines in rows.items()
    }

    counts = {short: len(lines) for short, lines in cleaned.items()}
    if len(set(counts.values())) != 1:
        raise PreprocessError(f"{label} not parallel: {counts}")

    target.mkdir(parents=True, exist_ok=True)
    for short, lines in cleaned.items():
        file_path = target / f"{short}.txt"
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{label}: {next(iter(counts.values()))} rows x {len(cleaned)} languages -> {target}")


def build_parser() -> argparse.ArgumentParser:
    """Parse --corpus, --out, and --norm."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw FLORES-200 / MASSIVE into the A1 eval corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  preprocess.py --corpus both\n"
            "  preprocess.py --corpus flores\n"
            "  preprocess.py --corpus massive --out /data/eval\n"
            "  preprocess.py --corpus both --norm nfkc\n"
            "\n"
            "Requires raw data from download_corpus.py first. Pipeline: strip edge\n"
            "whitespace, Unicode-normalize (--norm, default nfc), collapse\n"
            "whitespace runs; MASSIVE is limited to the test partition. Output layout:\n"
            "  <out>/flores/{dev,devtest}/<lang>.txt\n"
            "  <out>/massive/<lang>.txt\n"
            "where <lang> is one of: eng hin kan tam tel mal"
        ),
    )

    parser.add_argument(
        "--corpus",
        choices=["flores", "massive", "sample", "both", "all"],
        required=True,
        help='which corpus to preprocess: "flores" (writes dev + devtest splits), '
             '"massive" (test partition only), "sample" (starter-kit sample), '
             '"both" (flores + massive), or "all"',
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for the preprocessed .txt files; defaults to "
             "partA/corpus/preprocessed_corpus. Example: --out /data/eval",
    )

    parser.add_argument(
        "--norm",
        choices=["nfc", "nfkc"],
        default="nfc",
        help="Unicode normalization form: \"nfc\" (default; composes, preserves "
             "superscripts/fractions like km²) or \"nfkc\" (also compatibility-"
             "decomposes: km² -> km2, 24½ -> 241⁄2; changes ~10 devtest lines)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    norm = args.norm.upper()  # lowercase CLI choice -> unicodedata's "NFC"/"NFKC"

    try:
        if args.corpus in ("flores", "both", "all"):
            for split in FLORES_SPLITS:
                preprocess(load_flores(split), args.out / "flores" / split, f"flores/{split}", norm)

        if args.corpus in ("massive", "both", "all"):
            preprocess(load_massive(), args.out / "massive", f"massive/{MASSIVE_PARTITION}", norm)

        if args.corpus in ("sample", "all"):
            preprocess(load_sample(), args.out / "sample", "sample", norm)

    except PreprocessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
