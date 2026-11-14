#!/usr/bin/env python3
"""compression / denominator choice — evaluating 5 candidate denominators.

Denominators evaluated:
1. Tokens/byte               — Encoding-relative efficiency
2. Tokens/codepoint          — Unicode-level fragmentation
3. Tokens/grapheme           — User-perceived character-level fragmentation
4. Tokens/word               — Subword fertility
5. Tokens/parallel segment   — Cross-language token cost

Run: python partA/run_experiments.py --experiment compression
"""

import regex as re
from experiments import LANG_LABEL, save_fig

RESULT = "h6_compression"

GRAPHEME_RE = re.compile(r"\X")


def analyze(lines, encode):
    """Compute micro-averaged tokens across all 5 candidate denominators."""
    n = len(lines)
    tot_tokens = tot_bytes = tot_cp = tot_graphemes = tot_words = 0

    for line in lines:
        line = line.lower()
        toks = encode(line)
        nt = len(toks)
        b = len(line.encode("utf-8"))
        cp = len(line)
        gr = len(GRAPHEME_RE.findall(line))
        w = len(line.split())

        tot_tokens += nt
        tot_bytes += b
        tot_cp += cp
        tot_graphemes += gr
        tot_words += w

    return {
        "n": n,
        "tot_tokens": tot_tokens,
        "tot_bytes": tot_bytes,
        "tot_cp": tot_cp,
        "tot_graphemes": tot_graphemes,
        "tot_words": tot_words,
        "tok_per_byte": tot_tokens / tot_bytes,
        "tok_per_codepoint": tot_tokens / tot_cp,
        "tok_per_grapheme": tot_tokens / tot_graphemes,
        "tok_per_word": tot_tokens / tot_words,
        "tok_per_segment": tot_tokens / n,
        "bytes_per_token": tot_bytes / tot_tokens,
    }


def run(corpus: dict, encoders: dict):
    """Print the 5-denominator comparison table; return (payload, csv_rows)."""
    print("\ndenominators — tokenization efficiency across 5 candidate denominators")
    print("Evaluating:")
    print("  Tokens/byte              -> Encoding-relative efficiency")
    print("  Tokens/codepoint         -> Unicode-level fragmentation")
    print("  Tokens/grapheme          -> User-perceived character-level fragmentation")
    print("  Tokens/word              -> Subword fertility")
    print("  Tokens/parallel segment  -> Cross-language token cost")

    all_rows: list[dict] = []
    results: dict = {}
    ek = next((lang for lang in corpus if "eng" in lang), None)

    for tok, encode in encoders.items():
        per_lang: dict[str, dict] = {}
        rows: list[dict] = []

        for lang, lines in corpus.items():
            m = analyze(lines, encode)
            row = {
                "tokenizer": tok,
                "lang": lang,
                "language": LANG_LABEL.get(lang, lang),
                "n_sentences": m["n"],
                "tot_tokens": m["tot_tokens"],
                "tot_bytes": m["tot_bytes"],
                "tot_codepoints": m["tot_cp"],
                "tot_graphemes": m["tot_graphemes"],
                "tot_words": m["tot_words"],
                "tok_per_byte": m["tok_per_byte"],
                "tok_per_codepoint": m["tok_per_codepoint"],
                "tok_per_grapheme": m["tok_per_grapheme"],
                "tok_per_word": m["tok_per_word"],
                "tok_per_segment": m["tok_per_segment"],
                "bytes_per_token": m["bytes_per_token"],
            }
            rows.append(row)
            per_lang[lang] = row

        eng = per_lang.get(ek, rows[0])
        for r in rows:
            r["ratio_tpb"] = r["tok_per_byte"] / eng["tok_per_byte"]
            r["ratio_tpc"] = r["tok_per_codepoint"] / eng["tok_per_codepoint"]
            r["ratio_tpg"] = r["tok_per_grapheme"] / eng["tok_per_grapheme"]
            r["ratio_tpw"] = r["tok_per_word"] / eng["tok_per_word"]
            r["ratio_tps"] = r["tok_per_segment"] / eng["tok_per_segment"]
            all_rows.append(r)

        results[tok] = {"languages": per_lang}

        print(f"\n[{tok}]")
        hdr = (
            f"{'language':<11} "
            f"{'tok/byte':>10}  {'tok/cp':>10}  {'tok/graph':>11}  "
            f"{'tok/word':>10}  {'tok/seg':>10}   "
            f"{'bytes/tok':>10}"
        )
        sub = (
            f"{'':<11} "
            f"{'(encoding)':>10}  {'(unicode)':>10}  {'(perceived)':>11}  "
            f"{'(fertility)':>10}  {'(cross-lang)':>10}   "
            f"{'(density)':>10}"
        )
        print(hdr)
        print(sub)
        print("-" * len(hdr))

        for r in rows:
            print(
                f"{r['language']:<11} "
                f"{r['tok_per_byte']:>10.4f}  {r['tok_per_codepoint']:>10.4f}  {r['tok_per_grapheme']:>11.4f}  "
                f"{r['tok_per_word']:>10.4f}  {r['tok_per_segment']:>10.2f}   "
                f"{r['bytes_per_token']:>10.2f}"
            )

        print("\n  Ratios to English:")
        for r in rows:
            if r["lang"] == ek:
                continue
            print(
                f"    {r['language']:<10}: "
                f"tok/byte {r['ratio_tpb']:.2f}x | "
                f"tok/cp {r['ratio_tpc']:.2f}x | "
                f"tok/graph {r['ratio_tpg']:.2f}x | "
                f"tok/word {r['ratio_tpw']:.2f}x | "
                f"tok/seg {r['ratio_tps']:.2f}x"
            )

    return {"tokenizers": list(encoders), "results": results}, all_rows


def plot(payload):
    """Plot the 5 denominators across languages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    toks = payload["tokenizers"]
    metrics = [
        ("tok_per_byte", "Tokens / Byte\n(Encoding)"),
        ("tok_per_codepoint", "Tokens / Codepoint\n(Unicode)"),
        ("tok_per_grapheme", "Tokens / Grapheme\n(Perceived)"),
        ("tok_per_word", "Tokens / Word\n(Fertility)"),
        ("tok_per_segment", "Tokens / Segment\n(Cross-language)"),
    ]

    fig, axes = plt.subplots(len(toks), len(metrics), figsize=(3.2 * len(metrics), 3.5 * len(toks)), squeeze=False)

    for i, tok in enumerate(toks):
        res = payload["results"][tok]
        langs = list(res["languages"])
        x = np.arange(len(langs))
        for j, (key, title) in enumerate(metrics):
            ax = axes[i, j]
            vals = [res["languages"][l][key] for l in langs]
            ax.bar(x, vals, color="#2A9D8F", width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([LANG_LABEL.get(l, l) for l in langs], rotation=45, fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=8, weight="bold")
            if j == 0:
                ax.set_ylabel(tok, fontsize=8, weight="bold")
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Denominators: Comparing 5 granularities of tokenization efficiency",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_h6_compression.png")
