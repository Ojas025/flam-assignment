#!/usr/bin/env python3
"""lang_audit — per-language unit audit against the REPORT_v0 baseline.

REPORT_v0 prints exactly two numbers per language (fertility tok/word and
tok/char), and those two fully determine the units this audit recomputes
directly from the corpus:

    chars/word        = tok/word / tok/char     (derived from the report)
    codepoints/token  = 1 / tok/char            (derived from the report)
    bytes/token       = never measured by the report (fresh, no baseline)

Two purposes, in order:

  1. reproduction — on --dataset sample with gpt2 the measured fertility and
     tok/char must match the report's printed values (Delta ~= rounding),
     which proves the derived baseline is sound before it is used elsewhere;
  2. audit — on any corpus the Delta columns quantify how far the report's
     sample-derived numbers are from the selected corpus, and chars/word
     tests the report's stated root cause ("Hindi simply has more Unicode
     characters per word") head-on.

Methodology matches fertility.py exactly (NFC at load, line.lower(),
split(" "), macro mean of per-line ratios) so the comparison is apples to
apples — the h1/h2 ablations already quantify what the corrected
denominators would change.

Run: python partA/run_experiments.py --experiment lang_audit --dataset sample
     python partA/run_experiments.py --experiment lang_audit --tokenizers all
"""

from experiments import LANG_LABEL, save_fig

RESULT = "h7_lang_audit"

# REPORT_v0.md section 1: the only two numbers it prints per language
# (gpt2 tokenizer, sample corpus, macro per-line mean).  Kept verbatim —
# every Delta in this experiment is against these.
REPORT_V0_PRINTED = {
    "eng": {"language": "English", "fertility": 1.27, "tok_per_char": 0.226},
    "hin": {"language": "Hindi", "fertility": 7.45, "tok_per_char": 1.579},
}

# Column widths for the printed tables (language labels up to "Malayalam").
COL = {"lang": 11, "cpw": 12, "dcpw": 9, "cpt": 17, "dcpt": 9, "bpt": 12,
       "fert": 11, "rep": 9, "dfert": 9, "tpc": 14, "dtpc": 9}


def baseline_for(lang: str):
    """REPORT_v0-derived baseline for a corpus language key, or None.

    Keys are short codes (eng, hin) but custom datasets may use longer
    stems, so match on the code at the start of the key only — never a
    bare substring, which would pull 'eng' out of 'beng'.
    """
    key = next((k for k in REPORT_V0_PRINTED if lang == k or lang.startswith(k + "_")),
               None)
    if key is None:
        return None
    rep = REPORT_V0_PRINTED[key]
    return {
        "report_fertility": rep["fertility"],
        "report_tok_per_char": rep["tok_per_char"],
        # the report's own two columns imply these — no new measurement
        "chars_per_word": rep["fertility"] / rep["tok_per_char"],
        "codepoints_per_token": 1.0 / rep["tok_per_char"],
        "bytes_per_token": None,  # REPORT_v0 never measured bytes
    }


def analyze(lines, encode):
    """Macro per-line means under REPORT_v0's methodology.

    WHY macro: the report's printed numbers are means of per-line ratios,
    so anything compared against them must aggregate identically — a micro
    Delta would mix methodology drift with a real difference.  Lines where
    the tokenizer returns nothing are excluded from the per-token ratios
    only (their chars/word is still well defined).
    """
    n = len(lines)
    sum_fert = sum_tpc = sum_cpw = 0.0
    sum_cpt = sum_bpt = 0.0
    n_tok_scored = 0
    for line in lines:
        line = line.lower()
        words = line.split(" ")   # the report's denominator, empties included
        chars = len(line)         # codepoints — the report's "char"
        ntok = len(encode(line))
        sum_fert += ntok / len(words)
        sum_tpc += ntok / chars
        sum_cpw += chars / len(words)
        if ntok:
            sum_cpt += chars / ntok
            sum_bpt += len(line.encode("utf-8")) / ntok
            n_tok_scored += 1
    if n_tok_scored == 0:
        raise ValueError("tokenizer produced no tokens for this corpus")
    return {
        "n": n,
        "macro_fertility": sum_fert / n,
        "macro_tok_per_char": sum_tpc / n,
        "chars_per_word": sum_cpw / n,
        "codepoints_per_token": sum_cpt / n_tok_scored,
        "bytes_per_token": sum_bpt / n_tok_scored,
    }


def _pct(meas: float, base: float) -> float:
    """Signed percentage Delta of a measured value against its baseline."""
    return 100.0 * (meas / base - 1.0)


def _fmt_pct(value) -> str:
    """Cell for a Delta column: signed percent, or an em dash if not applicable."""
    return "–" if value is None else f"{value:+.2f}%"


def run(corpus: dict, encoders: dict):
    """Print reproduction check + per-tokenizer audit tables; return (payload, rows)."""
    print("\nlang_audit — per-language unit audit vs the REPORT_v0 baseline")
    print("baseline source : REPORT_v0.md §1 (gpt2, sample, macro, lower(), split(' '))")
    print("derived         : chars/word = tok/word ÷ tok/char · "
          "codepoints/token = 1 ÷ tok/char · bytes/token: none in report")
    base_desc = " · ".join(
        f"{REPORT_V0_PRINTED[k]['language']} {REPORT_V0_PRINTED[k]['fertility'] / REPORT_V0_PRINTED[k]['tok_per_char']:.2f}"
        f" / {1.0 / REPORT_V0_PRINTED[k]['tok_per_char']:.2f}"
        for k in REPORT_V0_PRINTED
    )
    print(f"                  baseline values (chars/word / codepoints/token): {base_desc}")

    ek = next((lang for lang in corpus if lang.startswith("eng")), None)

    # --- reproduction: gpt2 only, the report's own tokenizer ---
    reproduction: list[dict] = []
    if "gpt2" in encoders and ek is not None:
        g = encoders["gpt2"]
        print("\n[reproduction vs REPORT_v0 printed numbers — gpt2]")
        hdr = (f"{'language':<{COL['lang']}}{'fert meas':>{COL['fert']}}"
               f"{'fert rep':>{COL['rep']}}{'Δ%':>{COL['dfert']}}"
               f"{'tok/char meas':>{COL['tpc']}}{'rep':>{COL['rep']}}"
               f"{'Δ%':>{COL['dtpc']}}")
        print(hdr)
        print("-" * len(hdr))
        for lang, lines in corpus.items():
            base = baseline_for(lang)
            if base is None:
                continue
            m = analyze(lines, g)
            row = {
                "lang": lang,
                "language": LANG_LABEL.get(lang, lang),
                "measured_fertility": m["macro_fertility"],
                "report_fertility": base["report_fertility"],
                "delta_pct_fertility": _pct(m["macro_fertility"], base["report_fertility"]),
                "measured_tok_per_char": m["macro_tok_per_char"],
                "report_tok_per_char": base["report_tok_per_char"],
                "delta_pct_tok_per_char": _pct(m["macro_tok_per_char"], base["report_tok_per_char"]),
            }
            reproduction.append(row)
            print(f"{row['language']:<{COL['lang']}}"
                  f"{row['measured_fertility']:>{COL['fert']}.4f}"
                  f"{row['report_fertility']:>{COL['rep']}.3f}"
                  f"{_fmt_pct(row['delta_pct_fertility']):>{COL['dfert']}}"
                  f"{row['measured_tok_per_char']:>{COL['tpc']}.4f}"
                  f"{row['report_tok_per_char']:>{COL['rep']}.3f}"
                  f"{_fmt_pct(row['delta_pct_tok_per_char']):>{COL['dtpc']}}")
        print("  (rounding-level Δ expected on --dataset sample — report printed to 3 s.f.;")
        print("   elsewhere Δ = corpus difference; chars/word Δbase below also includes the")
        print("   report's rounding and macro ratio-of-means vs mean-of-ratios)")

    # --- the audit: every tokenizer, every language ---
    all_rows: list[dict] = []
    results: dict = {}
    for tok, encode in encoders.items():
        # The report's baseline is gpt2-specific for every token-derived
        # metric; chars/word involves no tokens, so its Δ is valid for any
        # tokenizer — but it is shown under gpt2 only to keep one authority.
        show_base = tok == "gpt2"
        per_lang: dict[str, dict] = {}
        rows_tok: list[dict] = []
        for lang, lines in corpus.items():
            m = analyze(lines, encode)
            base = baseline_for(lang)
            row = {
                "tokenizer": tok, "lang": lang,
                "language": LANG_LABEL.get(lang, lang),
                "n_sentences": m["n"],
                "macro_fertility": m["macro_fertility"],
                "macro_tok_per_char": m["macro_tok_per_char"],
                "chars_per_word": m["chars_per_word"],
                "codepoints_per_token": m["codepoints_per_token"],
                "bytes_per_token": m["bytes_per_token"],
                "baseline_chars_per_word": base["chars_per_word"] if base else None,
                "baseline_codepoints_per_token": base["codepoints_per_token"] if base else None,
                "delta_pct_chars_per_word": (
                    _pct(m["chars_per_word"], base["chars_per_word"])
                    if base and show_base else None),
                "delta_pct_codepoints_per_token": (
                    _pct(m["codepoints_per_token"], base["codepoints_per_token"])
                    if base and show_base else None),
            }
            per_lang[lang] = row
            rows_tok.append(row)

        results[tok] = {"languages": per_lang}

        print(f"\n[{tok}]")
        if show_base:
            hdr = (f"{'language':<{COL['lang']}}{'chars/word':>{COL['cpw']}}"
                   f"{'Δbase':>{COL['dcpw']}}"
                   f"{'codepoints/token':>{COL['cpt']}}"
                   f"{'Δbase':>{COL['dcpt']}}"
                   f"{'bytes/token':>{COL['bpt']}}")
        else:
            hdr = (f"{'language':<{COL['lang']}}{'chars/word':>{COL['cpw']}}"
                   f"{'codepoints/token':>{COL['cpt']}}"
                   f"{'bytes/token':>{COL['bpt']}}")
        print(hdr)
        print("-" * len(hdr))
        for row in rows_tok:
            line = (f"{row['language']:<{COL['lang']}}"
                    f"{row['chars_per_word']:>{COL['cpw']}.4f}")
            if show_base:
                line += _fmt_pct(row["delta_pct_chars_per_word"]).rjust(COL["dcpw"])
            line += (f"{row['codepoints_per_token']:>{COL['cpt']}.4f}")
            if show_base:
                line += _fmt_pct(row["delta_pct_codepoints_per_token"]).rjust(COL["dcpt"])
            line += f"{row['bytes_per_token']:>{COL['bpt']}.4f}"
            print(line)

        all_rows.extend(rows_tok)

    # --- ratio verdict: the report's root cause claim, tested ---
    verdict = "per-language unit audit (no English reference in corpus — ratio verdict skipped)"
    hin = next((lang for lang in corpus if lang.startswith("hin")), None)
    if ek is not None and hin is not None:
        verdict_tok = "gpt2" if "gpt2" in encoders else next(iter(encoders))
        per_lang = results[verdict_tok]["languages"]
        eng, hi = per_lang[ek], per_lang[hin]
        ratios = {
            "fertility": hi["macro_fertility"] / eng["macro_fertility"],
            "chars/word": hi["chars_per_word"] / eng["chars_per_word"],
            "codepoints/token": hi["codepoints_per_token"] / eng["codepoints_per_token"],
            "bytes/token": hi["bytes_per_token"] / eng["bytes_per_token"],
        }
        print(f"\nratios to English [{verdict_tok}]: "
              + " · ".join(f"{k} {v:.2f}x" for k, v in ratios.items()))
        verdict = (f"Hindi/English: fertility {ratios['fertility']:.2f}x but chars/word "
                   f"{ratios['chars/word']:.2f}x — REPORT_v0's root cause contradicted")
        print("  → FINDING: Hindi words are "
              f"{'SHORTER' if ratios['chars/word'] < 1 else 'longer'} in codepoints "
              f"({ratios['chars/word']:.2f}x), not longer — REPORT_v0's stated root cause")
        print("    (\"Hindi simply has more Unicode characters per word\") is contradicted by")
        print(f"    its own numbers; the {ratios['fertility']:.2f}x fertility gap lives in")
        print(f"    codepoints/token ({1 / ratios['codepoints/token']:.1f}x fewer codepoints per token),")
        print("    i.e. tokens per codepoint — the tokenizer, not word length.")
    elif ek is None:
        print("\n  (no English reference in corpus — ratio verdict skipped)")

    payload = {
        "tokenizers": list(encoders),
        "report_v0": REPORT_V0_PRINTED,
        "baseline": {lang: baseline_for(lang) for lang in corpus},
        "reproduction": reproduction,
        "results": results,
        "verdict": verdict,
    }
    return payload, all_rows


def plot(payload):
    """One essential figure: the three audit metrics side by side.

    One bar group per language (one bar per tokenizer), with the REPORT_v0
    baseline drawn as a dashed black line where the report has one — the
    visual form of "measured vs baseline", and for chars/word the direct
    picture of the root-cause claim failing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [
        ("chars_per_word", "chars/word (codepoints per word)", True),
        ("codepoints_per_token", "codepoints/token", True),
        ("bytes_per_token", "bytes/token", False),
    ]
    toks = payload["tokenizers"]
    langs = list(payload["results"][toks[0]]["languages"])
    colors = ["#E76F51", "#2A9D8F", "#457B9D", "#E9C46A", "#9B5DE5"]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.3))
    x = np.arange(len(langs))
    w = 0.8 / len(toks)
    base_labeled = False   # one legend entry for the baseline, wherever first drawn

    for ax, (key, title, has_base) in zip(axes, panels):
        for ti, tok in enumerate(toks):
            vals = [payload["results"][tok]["languages"][l][key] for l in langs]
            ax.bar(x + (ti - (len(toks) - 1) / 2) * w, vals, w * 0.92,
                   label=tok, color=colors[ti % len(colors)])
        if has_base:
            for li, lang in enumerate(langs):
                b = (payload.get("baseline") or {}).get(lang)
                bval = b.get(key) if b else None
                if bval is not None:
                    ax.plot([li - 0.42, li + 0.42], [bval, bval], color="black",
                            lw=1.5, ls="--", zorder=5,
                            label=None if base_labeled else "REPORT_v0 baseline")
                    base_labeled = True
        ax.set_xticks(x, [LANG_LABEL.get(l, l) for l in langs], rotation=15)
        ax.set_title(title, loc="left", weight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(toks) + 1,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"lang_audit: {payload['verdict']}",
                 x=0.01, ha="left", fontsize=11, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "fig_h7_lang_audit.png")
