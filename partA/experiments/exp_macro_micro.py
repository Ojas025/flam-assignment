#!/usr/bin/env python3
"""macro_micro — averaging ablation: macro mean vs micro sum/sum.

Hypothesis: macro averaging overweights short lines and distorts corpus-level
estimates; micro aggregation pools tokens and units directly.

Run: python partA/run_experiments.py --experiment metric_choice
"""

from experiments import LANG_LABEL, save_fig

RESULT = "h3_macro_micro"


def analyze(lines, encode):
    """Compute per-line macro ratios and pooled micro totals."""
    n = len(lines)
    m_tpc = m_tpb = m_fert = 0.0
    tot_tok = tot_chars = tot_bytes = tot_words = 0

    for line in lines:
        line = line.lower()
        toks = encode(line)
        nt = len(toks)
        cp = len(line)
        b = len(line.encode("utf-8"))
        words = line.split(" ")
        nw = len(words)

        m_tpc += nt / cp
        m_tpb += nt / b
        m_fert += nt / nw

        tot_tok += nt
        tot_chars += cp
        tot_bytes += b
        tot_words += nw

    mac_tpc, mic_tpc = m_tpc / n, tot_tok / tot_chars
    # With one segment per line, macro TPS is the mean per-line token count
    # and micro TPS is pooled tokens over pooled segments, so they coincide.
    mac_tps, mic_tps = tot_tok / n, tot_tok / n
    mac_tpb, mic_tpb = m_tpb / n, tot_tok / tot_bytes
    mac_fert, mic_fert = m_fert / n, tot_tok / tot_words

    return {
        "n": n,
        "macro_tpc": mac_tpc,
        "micro_tpc": mic_tpc,
        "tpc_delta_pct": 100.0 * (mic_tpc / mac_tpc - 1.0),
        "macro_tps": mac_tps,
        "micro_tps": mic_tps,
        "tps_delta_pct": 100.0 * (mic_tps / mac_tps - 1.0),
        "macro_tpb": mac_tpb,
        "micro_tpb": mic_tpb,
        "tpb_delta_pct": 100.0 * (mic_tpb / mac_tpb - 1.0),
        "macro_fertility": mac_fert,
        "micro_fertility": mic_fert,
        "fert_delta_pct": 100.0 * (mic_fert / mac_fert - 1.0),
    }


def run(corpus: dict, encoders: dict):
    """Print the macro vs micro table; return (payload, csv_rows)."""
    print("\nmacro_micro — aggregation ablation: macro (mean of line ratios) vs micro (sum / sum)")
    print("Hypothesis: macro overweights short lines; micro reflects true corpus-level token volume.")

    all_rows: list[dict] = []
    results: dict = {}
    ek = next((lang for lang in corpus if "eng" in lang), None)

    for tok, encode in encoders.items():
        per_lang: dict[str, dict] = {}
        rows_tok: list[dict] = []

        for lang, lines in corpus.items():
            m = analyze(lines, encode)
            row = {
                "tokenizer": tok,
                "lang": lang,
                "language": LANG_LABEL.get(lang, lang),
                "n_sentences": m["n"],
                "macro_tpc": m["macro_tpc"],
                "micro_tpc": m["micro_tpc"],
                "tpc_delta_pct": m["tpc_delta_pct"],
                "macro_tps": m["macro_tps"],
                "micro_tps": m["micro_tps"],
                "tps_delta_pct": m["tps_delta_pct"],
                "macro_tpb": m["macro_tpb"],
                "micro_tpb": m["micro_tpb"],
                "tpb_delta_pct": m["tpb_delta_pct"],
                "macro_fertility": m["macro_fertility"],
                "micro_fertility": m["micro_fertility"],
                "fert_delta_pct": m["fert_delta_pct"],
            }
            per_lang[lang] = row
            rows_tok.append(row)

        eng = per_lang.get(ek, rows_tok[0])
        for r in rows_tok:
            r["ratio_macro"] = r["macro_fertility"] / eng["macro_fertility"]
            r["ratio_micro"] = r["micro_fertility"] / eng["micro_fertility"]
            r["ratio_delta_pct"] = 100.0 * (r["ratio_micro"] / r["ratio_macro"] - 1.0)
            all_rows.append(r)

        results[tok] = {"languages": per_lang}

        print(f"\n[{tok}]")
        hdr = (
            f"{'language':<12} "
            f"{'macro tpc':>10}  {'micro tpc':>10}  {'tpcΔ%':>7}   "
            f"{'macro tps':>10}  {'micro tps':>10}  {'tpsΔ%':>7}   "
            f"{'macro tpb':>10}  {'micro tpb':>10}  {'tpbΔ%':>7}   "
            f"{'macro fert':>10}  {'micro fert':>10}  {'fertΔ%':>7}"
        )
        print(hdr)
        print("-" * len(hdr))

        for r in rows_tok:
            print(
                f"{r['language']:<12} "
                f"{r['macro_tpc']:>10.4f}  {r['micro_tpc']:>10.4f}  {r['tpc_delta_pct']:>+6.2f}%   "
                f"{r['macro_tps']:>10.2f}  {r['micro_tps']:>10.2f}  {r['tps_delta_pct']:>+6.2f}%   "
                f"{r['macro_tpb']:>10.4f}  {r['micro_tpb']:>10.4f}  {r['tpb_delta_pct']:>+6.2f}%   "
                f"{r['macro_fertility']:>10.4f}  {r['micro_fertility']:>10.4f}  {r['fert_delta_pct']:>+6.2f}%"
            )

        worst_fert = max(rows_tok, key=lambda x: abs(x["fert_delta_pct"]))
        worst_tpc = max(rows_tok, key=lambda x: abs(x["tpc_delta_pct"]))
        eng_row = next((x for x in rows_tok if "eng" in x["lang"]), None)
        eng_txt = f", eng fertΔ {eng_row['fert_delta_pct']:+.2f}%" if eng_row else ""
        print(
            f"  → max |fertΔ| = {abs(worst_fert['fert_delta_pct']):.2f}% ({worst_fert['language']}){eng_txt}; "
            f"max |tpcΔ| = {abs(worst_tpc['tpc_delta_pct']):.2f}% ({worst_tpc['language']}); "
            f"tpsΔ = 0.00% (identical by definition)"
        )

    return {"tokenizers": list(encoders), "results": results}, all_rows


def plot(payload):
    """Plot macro vs micro fertility across languages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arm_a, arm_b = "#E76F51", "#2A9D8F"
    toks = payload["tokenizers"]
    fig, axes = plt.subplots(1, len(toks), figsize=(5.5 * len(toks), 4.5), sharey=True)
    if len(toks) == 1:
        axes = [axes]
    for ax, tok in zip(axes, toks):
        rows = list(payload["results"][tok]["languages"].values())
        x, w = np.arange(len(rows)), 0.38
        ax.bar(x - w / 2, [r["macro_fertility"] for r in rows], w, label="macro (line mean)", color=arm_a)
        ax.bar(x + w / 2, [r["micro_fertility"] for r in rows], w, label="micro (sum/sum)", color=arm_b)
        for xi, r in zip(x, rows):
            ax.text(xi, max(r["macro_fertility"], r["micro_fertility"]) * 1.04,
                    f"{r['fert_delta_pct']:+.2f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [r["language"] for r in rows], rotation=15)
        ax.set_yscale("log")
        ax.set_title(tok, loc="left", weight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_ylabel("Fertility (tok/word), log scale")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("macro_micro: macro mean overweights short lines; micro is true corpus aggregate",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig_h3_fertility.png")
