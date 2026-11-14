#!/usr/bin/env python3
"""tok_dependence — tokenizer dependence of the fertility gap.

Claim under test: 'Hindi has more Unicode chars/word, so any tokenizer
struggles.'  Ablation: hold corpus and preprocessing fixed and change only
the tokenizer — the gap moves with the tokenizer while chars/word (the
tokenizer-independent control) does not.

Run: python partA/run_experiments.py --experiment tok_dependence
"""

import regex as re

from experiments import LANG_LABEL, save_fig

RESULT = "h5_tokenizer"

GRAPHEME_RE = re.compile(r"\X")


def analyze(lines, encode):
    """fertility.py's analyze plus tokenizer-independent unit ratios.

    chars/word and graphemes/word are computed from the corpus alone — they
    are the control that must stay constant across tokenizers.
    """
    fert = tpc = 0.0
    tot_tokens = tot_words = tot_chars = tot_graphemes = 0
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        words = line.split(" ")
        w = line.split()          # unit ratios use the corrected word count
        fert += len(tokens) / len(words)
        tpc += len(tokens) / len(line)
        tot_tokens += len(tokens)
        tot_words += len(w)
        tot_chars += len(line)
        tot_graphemes += len(GRAPHEME_RE.findall(line))
    n = len(lines)
    return {
        "n": n,
        "macro_fert": fert / n,
        "macro_tpc": tpc / n,
        "chars_per_word": tot_chars / tot_words,
        "graphemes_per_word": tot_graphemes / tot_words,
        "tokens_per_sentence": tot_tokens / n,
    }


def run(corpus: dict, encoders: dict):
    """Print the per-tokenizer ratio table + verdict; return (payload, csv_rows)."""
    print("\ntok_dependence — tokenizer dependence of the fertility gap")
    print("Claim: 'Hindi has more Unicode chars/word, so any tokenizer struggles.'")
    print("Ablation: hold corpus+preprocessing fixed, change only tokenizer.")

    all_rows: list[dict] = []
    results: dict = {}
    ek = next(lang for lang in corpus if "eng" in lang)

    for tok, encode in encoders.items():
        per_lang: dict[str, dict] = {}
        rows_tok: list[dict] = []
        for lang, lines in corpus.items():
            m = analyze(lines, encode)
            row = {
                "tokenizer": tok, "lang": lang,
                "language": LANG_LABEL.get(lang, lang),
                "n_sentences": m["n"],
                "macro_fertility": m["macro_fert"],
                "macro_tpc": m["macro_tpc"],
                "chars_per_word": m["chars_per_word"],
                "graphemes_per_word": m["graphemes_per_word"],
                "tokens_per_sentence": m["tokens_per_sentence"],
            }
            rows_tok.append(row)
            per_lang[lang] = row
            all_rows.append(row)
        results[tok] = {"languages": per_lang}

        eng = per_lang[ek]
        print(f"\n[{tok}] ratios to English")
        print(f"{'language':<10}{'fertility':>10}{'tok/char':>10}{'chars/word':>12}"
              f"{'fert ratio':>12}{'tpc ratio':>11}{'cpw ratio':>11}")
        print("-" * 66)
        for r in rows_tok:
            print(f"{r['language']:<10}{r['macro_fertility']:>10.4f}{r['macro_tpc']:>10.5f}"
                  f"{r['chars_per_word']:>12.2f}"
                  f"{r['macro_fertility'] / eng['macro_fertility']:>11.2f}x"
                  f"{r['macro_tpc'] / eng['macro_tpc']:>10.2f}x"
                  f"{r['chars_per_word'] / eng['chars_per_word']:>10.2f}x")

        non_eng = next(r for r in rows_tok if "eng" not in r["lang"])
        print(f"  → {non_eng['language']}/English: fert "
              f"{non_eng['macro_fertility'] / eng['macro_fertility']:.2f}x but chars/word "
              f"{non_eng['chars_per_word'] / eng['chars_per_word']:.2f}x (tokenizer-independent control)")

    return {"tokenizers": list(encoders), "results": results}, all_rows


def plot(payload):
    """One essential figure: fertility per language, one panel per tokenizer.

    English-relative ratios are annotated on the bars — the gap that moves
    when only the tokenizer changes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    bar_c = "#457B9D"
    toks = payload["tokenizers"]
    fig, axes = plt.subplots(1, len(toks), figsize=(5.5 * len(toks), 4.5), sharey=True)
    if len(toks) == 1:
        axes = [axes]
    for ax, tok in zip(axes, toks):
        rows = list(payload["results"][tok]["languages"].values())
        eng = next(r for r in rows if "eng" in r["lang"])
        x = np.arange(len(rows))
        ax.bar(x, [r["macro_fertility"] for r in rows], 0.6, color=bar_c)
        for xi, r in zip(x, rows):
            ratio = r["macro_fertility"] / eng["macro_fertility"]
            ax.text(xi, r["macro_fertility"] * 1.04, f"{ratio:.2f}x",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [r["language"] for r in rows], rotation=15)
        ax.set_yscale("log")
        ax.set_title(tok, loc="left", weight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.25, which="both")
    axes[0].set_ylabel("Fertility (tok/word), log scale")
    fig.suptitle("tok_dependence: fertility gap (× English) moves with the tokenizer",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig_h5_fertility.png")
