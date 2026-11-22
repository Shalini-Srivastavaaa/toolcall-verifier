"""Results chart for the README (D-032): AUROC with 95% bootstrap intervals on the 363 real test rows, read from the committed analysis file.

    .venv/bin/python scripts/make_results_chart.py            # writes docs/figures/auroc_comparison.png

Source: results/final_qwen_v3b/analysis_1_paired.txt (section "OVERALL": AUROC and 95% CI, 2,000 bootstrap resamples). Nothing is typed in by hand.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "final_qwen_v3b" / "analysis_1_paired.txt"
OUT = ROOT / "docs" / "figures" / "auroc_comparison.png"
ORDER = ["ModernBERT v3b", "Qwen2.5-1.5B", "GPT-5-mini", "GPT-5-nano", "tool name only", "length only"]
LABEL = {"ModernBERT v3b": "ModernBERT v3b\n(150M, laptop CPU)", "Qwen2.5-1.5B": "Qwen2.5-1.5B + LoRA\n(needs a GPU)", "GPT-5-mini": "GPT-5-mini judge\n(API)", "GPT-5-nano": "GPT-5-nano judge\n(API)",
         "tool name only": "tool name only\n(shortcut)", "length only": "length only\n(shortcut)"}
COLOR = {"ModernBERT v3b": "#1f4e9c", "Qwen2.5-1.5B": "#6b8fd1", "GPT-5-mini": "#b5651d", "GPT-5-nano": "#d9a679", "tool name only": "#8a8a8a", "length only": "#b8b8b8"}


def read_overall():
    text = SRC.read_text()
    block = text[text.index("OVERALL:"): text.index("NATURAL rows:")]
    out = {}
    for m in re.finditer(r"^\s+(ModernBERT v3b|Qwen2\.5-1\.5B|GPT-5-mini|GPT-5-nano|length only|tool name only)\s+AUROC ([0-9.]+)\s+\(95% CI ([0-9.]+)-([0-9.]+)\)", block, re.M):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    assert set(out) == set(ORDER), out.keys()
    return out


def main():
    d = read_overall()
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=170)
    ys = list(range(len(ORDER)))[::-1]
    for y, k in zip(ys, ORDER):
        a, lo, hi = d[k]
        ax.errorbar(a, y, xerr=[[a - lo], [hi - a]], fmt="o", color=COLOR[k], ecolor=COLOR[k], elinewidth=2.4, capsize=4, markersize=9, zorder=3)
        ax.text(hi + 0.008, y, f"{a:.3f}  ({lo:.3f}-{hi:.3f})", va="center", fontsize=9.5, color="#222")
    ax.axvline(0.5, color="#999", ls=":", lw=1)
    ax.set_ylim(-0.75, len(ORDER) - 0.4)
    ax.text(0.503, -0.62, "chance (0.5)", fontsize=8.5, color="#777", va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[k] for k in ORDER], fontsize=10)
    ax.set_xlim(0.45, 0.93)
    ax.set_xlabel("AUROC on the 363 human-reviewed real test rows (95% bootstrap interval, 2,000 resamples)", fontsize=9.5)
    ax.set_title("Detecting incorrect agent tool calls: AUROC with 95% intervals", fontsize=12, loc="left", pad=12)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT, {k: d[k] for k in ORDER})


if __name__ == "__main__":
    main()
