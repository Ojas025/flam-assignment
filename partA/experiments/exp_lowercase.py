#!/usr/bin/env python3
"""lowercase — lowercasing ablation: line.lower() vs case-preserved.

Hypothesis: lower() changes token counts for Latin-script text (GPT-2 BPE is
case-sensitive), but is a no-op for Indic scripts that lack case — making the
metric asymmetric across languages.

Run: python partA/run_experiments.py --experiment lowercase
"""

import unicodedata

from experiments import LANG_LABEL, save_fig

RESULT = "h2_lower"

def analyze(lines, encode):
    fert_lower = fert_raw = tpc_lower = tpc_raw = 0.0
    tokens_lower = tokens_raw = 0

    text_changed = token_changed = latin_upper = nonlatin_upper = 0
    for line in lines:

        lower = line.lower()
        ids_lower = encode(lower)
        ids_raw = encode(line)

        fert_lower += len(ids_lower) / len(lower.split(" "))
        fert_raw += len(ids_raw) / len(line.split(" "))

        tpc_lower += len(ids_lower) / len(lower)
        tpc_raw += len(ids_raw) / len(line)

        tokens_lower += len(ids_lower)
        tokens_raw += len(ids_raw)

        text_changed += lower != line
        token_changed += len(ids_lower) != len(ids_raw)

        for ch in line:
            if ch.isupper():
                if "LATIN" in unicodedata.name(ch, ""):
                    latin_upper += 1
                else:
                    nonlatin_upper += 1

    n = len(lines)

    return {
        "n": n,
        "text_changed": text_changed,
        "token_changed": token_changed,
        "latin_upper": latin_upper,
        "nonlatin_upper": nonlatin_upper,
        "tokens_lower": tokens_lower,
        "tokens_raw": tokens_raw,
        "fert_lower": fert_lower / n,
        "fert_raw": fert_raw / n,
        "tpc_lower": tpc_lower / n,
        "tpc_raw": tpc_raw / n,
    }


def pct(after: float, before: float) -> float:
    """Signed percentage change of `after` relative to `before`."""
    return 100.0 * (after / before - 1.0)


def run(corpus: dict, encoders: dict):
    """Print the ablation table + verdict; return (payload, csv_rows)."""
    print("\nlowercase — lowercasing ablation: line.lower() vs case-preserved")
    print("Hypothesis: lower() changes token counts for Latin-script text,")
    print("not for Indic scripts that lack case.")

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
                "text_changed": m["text_changed"],
                "text_changed_pct": 100.0 * m["text_changed"] / m["n"],
                "token_changed": m["token_changed"],
                "token_changed_pct": 100.0 * m["token_changed"] / m["n"],
                "latin_upper": m["latin_upper"],
                "nonlatin_upper": m["nonlatin_upper"],
                "tokens_lower": m["tokens_lower"],
                "tokens_raw": m["tokens_raw"],
                "token_delta_pct": pct(m["tokens_raw"], m["tokens_lower"]),
                "fert_lower": m["fert_lower"],
                "fert_raw": m["fert_raw"],
                "fert_delta_pct": pct(m["fert_raw"], m["fert_lower"]),
                "tpc_lower": m["tpc_lower"],
                "tpc_raw": m["tpc_raw"],
                "tpc_delta_pct": pct(m["tpc_raw"], m["tpc_lower"]),
            }

            rows_tok.append(row)
            per_lang[lang] = row
            all_rows.append(row)

        results[tok] = {"languages": per_lang}

        col_tok_lower, col_tok_raw = "tokens lower()", "tokens raw"
        col_fert_lower, col_fert_raw = "fert lower()", "fert raw"

        print(f"\n[{tok}]")
        print(f"{'language':<10}{'text-chg%':>10}{'tok-chg%':>9}{col_tok_lower:>15}{col_tok_raw:>12}"
              f"{'tokenΔ%':>10}{col_fert_lower:>13}{col_fert_raw:>11}{'fertΔ%':>10}{'tpcΔ%':>10}")
        print("-" * 100)

        for r in rows_tok:
            print(f"{r['language']:<10}{r['text_changed_pct']:>9.1f}{r['token_changed_pct']:>9.1f}"
                  f"{r['tokens_lower']:>15,}{r['tokens_raw']:>12,}{r['token_delta_pct']:>+10.4f}"
                  f"{r['fert_lower']:>13.4f}{r['fert_raw']:>11.4f}{r['fert_delta_pct']:>+10.4f}"
                  f"{r['tpc_delta_pct']:>+10.4f}")

        eng = next(r for r in rows_tok if "eng" in r["lang"])
        non_eng = [r for r in rows_tok if "eng" not in r["lang"]]
        max_non_eng = max((abs(r["token_delta_pct"]) for r in non_eng), default=0.0)

        print(f"  → eng: text-chg {eng['text_changed_pct']:.1f}%, tokenΔ {eng['token_delta_pct']:+.4f}%, "
              f"fertΔ {eng['fert_delta_pct']:+.4f}% | max non-eng tokenΔ {max_non_eng:.4f}%")

    return {"tokenizers": list(encoders), "results": results}, all_rows


def plot(payload):
    """One essential figure: both casing arms per language, Δ% annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arm_a, arm_b = "#E76F51", "#2A9D8F"   # line.lower() / case-preserved
    toks = payload["tokenizers"]
    fig, axes = plt.subplots(1, len(toks), figsize=(5.5 * len(toks), 4.5), sharey=True)
    if len(toks) == 1:
        axes = [axes]
    for ax, tok in zip(axes, toks):
        rows = list(payload["results"][tok]["languages"].values())
        x, w = np.arange(len(rows)), 0.38
        ax.bar(x - w / 2, [r["fert_lower"] for r in rows], w, label="line.lower()", color=arm_a)
        ax.bar(x + w / 2, [r["fert_raw"] for r in rows], w, label="case-preserved", color=arm_b)
        for xi, r in zip(x, rows):
            ax.text(xi, max(r["fert_lower"], r["fert_raw"]) * 1.04, f"{r['fert_delta_pct']:+.2f}%",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [r["language"] for r in rows], rotation=15)
        ax.set_yscale("log")
        ax.set_title(tok, loc="left", weight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_ylabel("Fertility (tok/word), log scale")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("lowercase: lower() changes Latin token counts, not Indic-script counts",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig_h2_fertility.png")
