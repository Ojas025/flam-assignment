#!/usr/bin/env python3
"""split_space — word split ablation: line.split(' ') vs line.split().

Hypothesis: split(' ') keeps empty strings from repeated spaces, inflating
the word denominator and deflating fertility; split() collapses them, so
affected lines report higher fertility.

Run: python partA/run_experiments.py --experiment split_space
"""

from experiments import LANG_LABEL, save_fig

RESULT = "h1_split_space"


def analyze(lines, encode):
    """fertility.py's analyze with one mutation: words = line.split().

    Every metric is accumulated per arm — `space` measures the split(' ')
    arm, `ws` the split() arm — so each reported column and delta is
    calculated from that arm's own numbers. tpc (tokens/char, matching
    fertility.py) is the control: neither arm rewrites the text, so the
    arms are expected to agree; that agreement is measured, not asserted.
    """
    fert_space = fert_ws = 0.0
    tpc_space = tpc_ws = 0.0
    words_space = words_ws = affected = 0
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        chars = len(line)
        ws = line.split(" ")
        w = line.split()
        fert_space += len(tokens) / len(ws)
        fert_ws += len(tokens) / len(w)
        tpc_space += len(tokens) / chars
        tpc_ws += len(tokens) / chars
        words_space += len(ws)
        words_ws += len(w)
        affected += len(ws) != len(w)
    n = len(lines)
    return {
        "n": n,
        "words_space": words_space,
        "words_ws": words_ws,
        "empty_words": words_space - words_ws,
        "affected": affected,
        "fert_space": fert_space / n,
        "fert_ws": fert_ws / n,
        "tpc_space": tpc_space / n,
        "tpc_ws": tpc_ws / n,
    }


def run(corpus: dict, encoders: dict):
    """Print the ablation table + verdict; return (payload, csv_rows)."""
    print("\nsplit_space — word split ablation: line.split(' ') vs line.split()")
    print("Hypothesis: split(' ') keeps empty strings from repeated spaces,")
    print("inflating the word count and deflating fertility.")

    all_rows: list[dict] = []
    results: dict = {}
    for tok, encode in encoders.items():
        per_lang: dict[str, dict] = {}
        rows_tok: list[dict] = []
        for lang, lines in corpus.items():
            m = analyze(lines, encode)
            row = {
                "tokenizer": tok, "lang": lang,
                "language": LANG_LABEL.get(lang, lang),
                "n_sentences": m["n"],
                "words_space": m["words_space"],
                "words_ws": m["words_ws"],
                "empty_words": m["empty_words"],
                "affected_lines": m["affected"],
                "affected_pct": 100.0 * m["affected"] / m["n"],
                "fert_space": m["fert_space"],
                "fert_ws": m["fert_ws"],
                "delta_abs": m["fert_ws"] - m["fert_space"],
                "delta_pct": 100.0 * (m["fert_ws"] / m["fert_space"] - 1.0),
                "tpc_space": m["tpc_space"],
                "tpc_ws": m["tpc_ws"],
                "tpc_delta_pct": 100.0 * (m["tpc_ws"] / m["tpc_space"] - 1.0),
            }
            rows_tok.append(row)
            per_lang[lang] = row
            all_rows.append(row)

        results[tok] = {"languages": per_lang}
        col_fert_sp, col_fert_ws = "fert split(' ')", "fert split()"
        col_words_sp, col_words_ws = "words split(' ')", "words split()"
        print(f"\n[{tok}]")
        print(f"{'language':<10}{col_fert_sp:>15}{col_fert_ws:>13}{'fertΔ%':>9}"
              f"{col_words_sp:>17}{col_words_ws:>14}{'empty':>8}{'affected%':>11}")
        print("-" * 96)
        for r in rows_tok:
            print(f"{r['language']:<10}{r['fert_space']:>15.4f}{r['fert_ws']:>13.4f}{r['delta_pct']:>+9.2f}"
                  f"{r['words_space']:>17,}{r['words_ws']:>14,}{r['empty_words']:>8,}{r['affected_pct']:>10.2f}")

        eng = next((r for r in rows_tok if "eng" in r["lang"]), None)
        worst = max(rows_tok, key=lambda r: abs(r["delta_pct"]))
        tpc_worst = max(rows_tok, key=lambda r: abs(r["tpc_delta_pct"]))
        eng_txt = f", eng {eng['delta_pct']:+.2f}%" if eng else ""
        print(f"  → max |fertΔ| = {abs(worst['delta_pct']):.2f}% ({worst['language']}){eng_txt}; "
              f"max |tpcΔ| = {abs(tpc_worst['tpc_delta_pct']):.4f}% (control: same tokens, same chars)")

    return {"tokenizers": list(encoders), "results": results}, all_rows


def plot(payload):
    """One essential figure: both split arms per language, Δ% annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arm_a, arm_b = "#E76F51", "#2A9D8F"   # artifact arm / corrected arm
    toks = payload["tokenizers"]
    fig, axes = plt.subplots(1, len(toks), figsize=(5.5 * len(toks), 4.5), sharey=True)
    if len(toks) == 1:
        axes = [axes]
    for ax, tok in zip(axes, toks):
        rows = list(payload["results"][tok]["languages"].values())
        x, w = np.arange(len(rows)), 0.38
        ax.bar(x - w / 2, [r["fert_space"] for r in rows], w, label="split(' ')", color=arm_a)
        ax.bar(x + w / 2, [r["fert_ws"] for r in rows], w, label="split()", color=arm_b)
        for xi, r in zip(x, rows):
            ax.text(xi, max(r["fert_space"], r["fert_ws"]) * 1.04, f"{r['delta_pct']:+.2f}%",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [r["language"] for r in rows], rotation=15)
        ax.set_yscale("log")
        ax.set_title(tok, loc="left", weight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_ylabel("Fertility (tok/word), log scale")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("split_space: split(' ') inflates the word denominator, deflating fertility",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig_h1_fertility.png")
