#!/usr/bin/env python3
"""arg_order — baseline = langs[0]: ratio and label depend on argument order.

Bug: fertility.py picks the first corpus as reference, so the reported ratio
and the 'worse/better' label flip when the argument order is reversed.  The
table shows every ordered (base, comparison) pair from the same numbers.

Run: python partA/run_experiments.py --experiment arg_order
"""

from experiments import LANG_LABEL, save_fig

RESULT = "h4_baseline"


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

    return {"tokenizers": list(encoders), "results": all_rows}, all_rows


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
