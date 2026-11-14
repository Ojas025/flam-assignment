"""Experiment modules for the partA fertility.py audit (Task A2).

Not standalone: partA/run_experiments.py loads the corpus and tokenizers,
then calls each module's run(corpus, encoders), which prints the essential
output and returns (payload, csv_rows) for the runner to save.  Each module
then renders its own single essential figure via plot(payload).
"""

from pathlib import Path

LANG_LABEL = {
    "eng": "English", "hin": "Hindi",
    "kan": "Kannada", "tam": "Tamil", "tel": "Telugu", "mal": "Malayalam",
    "eng_Latn": "English", "hin_Deva": "Hindi",
    "kan_Knda": "Kannada", "tam_Taml": "Tamil",
    "tel_Telu": "Telugu", "mal_Mlym": "Malayalam",
}

DPI = 170


def save_fig(fig, name: str) -> None:
    """Write a figure to partA/plots/ and report the repo-relative path.

    Shared by every experiment's plot() so all figures land in one place and
    are reported identically in the CLI output.
    """
    import matplotlib.pyplot as plt

    part_a = Path(__file__).resolve().parents[1]
    out = part_a / "plots" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(part_a.parent)}")
