#!/usr/bin/env python3
"""arg_order — baseline = langs[0]: ratio and label depend on argument order.

Bug: fertility.py picks the first corpus as reference, so the reported ratio
and the 'worse/better' label flip when the argument order is reversed.  The
table shows every ordered (base, comparison) pair from the same numbers.

Run: python partA/run_experiments.py --experiment arg_order
"""

import re

from experiments import LANG_LABEL, save_fig

RESULT = "h4_baseline"


def cli_ratio(lines_by_lang, tokenizer, order):
    """Run starter_kit/fertility.py with --corpus flags in the given order."""
    import subprocess, sys, tempfile
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for lang in order:
            (tmp / f"{lang}.txt").write_text("\n".join(lines_by_lang[lang]) + "\n", encoding="utf-8")
        cmd = [sys.executable, str(root / "starter_kit" / "fertility.py"), "--tokenizer", tokenizer]
        for lang in order:
            cmd += ["--corpus", f"{lang}={tmp / (lang + '.txt')}"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    match = re.search(r"([A-Za-z]+) is ([0-9.]+)x the fertility of ([A-Za-z]+)", out)
    if not match:
        raise RuntimeError(f"unparsed fertility output: {out!r}")
    return {"order": list(order), "ratio_parsed": float(match.group(2)),
            "verdict": match.group(1) + " vs " + match.group(3)}


def analyze(lines, encode):
    fert = 0.0
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        fert += len(tokens) / len(line.split(" "))
    return fert / len(lines)


def run(corpus: dict, encoders: dict):
    """Print the ordered-pair ratio table; return (payload, csv_rows)."""

    print("\narg_order — baseline = langs[0]: ratio and label depend on argument order")
    print("Bug: fertility.py picks the first corpus as reference, so the ratio")
    print("and 'worse/better' label flip when argument order is reversed.")

    all_rows: list[dict] = []
    for tok, encode in encoders.items():
        fert = {lang: analyze(lines, encode) for lang, lines in corpus.items()}

        rows_tok: list[dict] = []
        for base_lang, base_fert in fert.items():
            for cmp_lang, cmp_fert in fert.items():
                if base_lang == cmp_lang:
                    continue

                rows_tok.append({
                    "tokenizer": tok,
                    "base": LANG_LABEL.get(base_lang, base_lang),
                    "comparison": LANG_LABEL.get(cmp_lang, cmp_lang),
                    "base_fert": f"{base_fert:.4f}",
                    "cmp_fert": f"{cmp_fert:.4f}",
                    "ratio": f"{cmp_fert / base_fert:.4f}",
                    "verdict": "worse" if cmp_fert > base_fert else "better",
                })

        all_rows.extend(rows_tok)

        print(f"\n[{tok}] — ratio flips with argument order")
        print(f"{'base':<10}{'comparison':<12}{'base fert':>11}{'cmp fert':>10}{'ratio':>9}{'verdict':>9}")
        print("-" * 61)

        for r in rows_tok:
            print(f"{r['base']:<10}{r['comparison']:<12}{r['base_fert']:>11}{r['cmp_fert']:>10}"
                  f"{r['ratio']:>9}{r['verdict']:>9}")

        ek = next((lang for lang in corpus if "eng" in lang), None)
        other = next((lang for lang in corpus if not (ek and "eng" in lang)), None)

        if ek and other:
            fwd = fert[other] / fert[ek]
            print(f"  → same data, opposite verdicts: {LANG_LABEL[ek]} base: "
                  f"{LANG_LABEL[other]} is {fwd:.2f}x ({'worse' if fwd > 1 else 'better'}) ↔ "
                  f"{LANG_LABEL[other]} base: {LANG_LABEL[ek]} is {1 / fwd:.2f}x "
                  f"({'worse' if fwd < 1 else 'better'})")

    cli_runs = []
    reciprocity_in_memory = 0.0
    ek = next((lang for lang in corpus if "eng" in lang), None)
    other = next((lang for lang in corpus if not (ek and "eng" in lang)), None)
    if ek and other and encoders:
        tok0 = next(iter(encoders))
        fwd = fert[other] / fert[ek] if False else None
    # in-memory reciprocity over all pairs
    mem_errs = []
    for tok, encode in encoders.items():
        f = {lang: analyze(lines, encode) for lang, lines in corpus.items()}
        for base_lang in f:
            for cmp_lang in f:
                if base_lang == cmp_lang:
                    continue
                # pair reciprocity: ratio(base->cmp) * ratio(cmp->base) == 1
                pass
    # compute max in-memory error across tokenizer/ordered pairs
    mem_errs = []
    for tok, encode in encoders.items():
        f = {lang: analyze(lines, encode) for lang, lines in corpus.items()}
        for a in f:
            for b in f:
                if a == b:
                    continue
                mem_errs.append(abs(f[a] / f[b] * f[b] / f[a] - 1.0))
    reciprocity_in_memory = max(mem_errs) if mem_errs else 0.0

    if ek and other and encoders:
        tok0 = next(iter(encoders))
        for order in ([ek, other], [other, ek]):
            # map lang codes to corpus lang keys; cli_ratio expects lines_by_lang with those keys
            cli_runs.append(cli_ratio(corpus, tok0, order))

    payload = {"tokenizers": list(encoders), "results": all_rows,
               "cli_runs": cli_runs,
               "reciprocity_in_memory": reciprocity_in_memory,
               "reciprocity_printed": (cli_runs[0]["ratio_parsed"] * cli_runs[1]["ratio_parsed"]) if len(cli_runs) >= 2 else 0.0}
    print(f"→ CLI reciprocity: in-memory error {reciprocity_in_memory:.1e}, printed product {payload['reciprocity_printed']:.4f}")
    return payload, all_rows


def plot(payload):
    """One essential figure: the same pair's ratio under both argument orders."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arm_a, arm_b = "#E76F51", "#2A9D8F"   # English-as-base / other-as-base
    toks = payload["tokenizers"]
    rows_all = payload["results"]
    fig, axes = plt.subplots(1, len(toks), figsize=(5.5 * len(toks), 4.5), sharey=True)
    if len(toks) == 1:
        axes = [axes]
    for ax, tok in zip(axes, toks):
        rows_tok = [r for r in rows_all if r["tokenizer"] == tok]
        anchor = next((r["base"] for r in rows_tok if r["base"] == "English"), rows_tok[0]["base"])
        fwd = [r for r in rows_tok if r["base"] == anchor]          # anchor first
        rev = {r["base"]: r for r in rows_tok if r["comparison"] == anchor}  # anchor second
        others = [r["comparison"] for r in fwd]
        x, w = np.arange(len(others)), 0.38
        f_ratios = [float(r["ratio"]) for r in fwd]
        r_ratios = [float(rev[o]["ratio"]) for o in others]
        ax.bar(x - w / 2, f_ratios, w, label=f"{anchor} as baseline", color=arm_a)
        ax.bar(x + w / 2, r_ratios, w, label="reversed order", color=arm_b)
        for xi, f, r in zip(x, f_ratios, r_ratios):
            ax.text(xi, f * 1.04, f"{f:.2f}x", ha="center", va="bottom", fontsize=8)
            ax.text(xi, r * 1.04, f"{r:.2f}x", ha="center", va="bottom", fontsize=8)
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x, others, rotation=15)
        ax.set_yscale("log")
        ax.set_title(tok, loc="left", weight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_ylabel("Fertility ratio (log scale)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("arg_order: baseline = first language — the ratio flips with argument order",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig_h4_baseline.png")
