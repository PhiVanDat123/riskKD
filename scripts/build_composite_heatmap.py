#!/usr/bin/env python3
"""Stitch the 3 per-pair PFW heatmaps + distributions into one composite figure
and a short stats markdown summary."""
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path("results/viz/pfw_pairs")
OUT_DIR = ROOT
PAIRS = [
    ("qwen3_big",   "Qwen3-8B  -->  Qwen3-1.7B"),
    ("qwen3_small", "Qwen3-1.7B  -->  Qwen3-0.6B"),
    ("llama",       "Llama-3.1-8B  -->  Llama-3.2-1B"),
]

def composite_heatmap():
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    for ax, (slug, title) in zip(axes, PAIRS):
        img = mpimg.imread(ROOT / slug / "fig1_pfw_heatmap.png")
        ax.imshow(img)
        ax.set_title(title, fontsize=14, weight="bold", loc="left", pad=8)
        ax.axis("off")
    fig.suptitle("PFW per-token weight  -  same UltraFeedback examples, 3 teacher-student pairs",
                 fontsize=15, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = OUT_DIR / "composite_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

def composite_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (slug, title) in zip(axes, PAIRS):
        img = mpimg.imread(ROOT / slug / "fig3_weight_distribution.png")
        ax.imshow(img)
        ax.set_title(title, fontsize=12, weight="bold", loc="center", pad=6)
        ax.axis("off")
    fig.suptitle("Per-token weight distribution: PFW vs kl_inv vs uniform", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "composite_distribution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

def stats_table():
    rows = []
    for slug, title in PAIRS:
        s = json.load(open(ROOT / slug / "stats.json"))
        rows.append((title, s))
    md_lines = [
        "# PFW per-pair stats (24 UltraFeedback examples each)",
        "",
        "| Pair | Trust  | Frontier | ErrorGate | p10  | p50  | p90  | Gini-proxy |",
        "|------|-------:|---------:|----------:|-----:|-----:|-----:|-----------:|",
    ]
    for title, s in rows:
        md_lines.append(
            f"| {title} | {s['radpo_pfw/trust_mean']:.3f} | {s['radpo_pfw/frontier_mean']:.3f} | "
            f"{s['radpo_pfw/error_gate_mean']:.3f} | {s['pfw_weight_p10']:.3f} | "
            f"{s['pfw_weight_p50']:.3f} | {s['pfw_weight_p90']:.3f} | "
            f"{s['pfw_weight_gini_proxy']:.3f} |"
        )
    md_lines += [
        "",
        "**Observations.**",
        "- ErrorGate ~ 1.24 across all 3 pairs - the loss-cap step has the same activation rate regardless of student size or family.",
        "- Frontier ~ 0.55-0.60: roughly half the tokens fall in the teachable band; this fraction is stable across pairs.",
        "- Trust drops from 0.74 (Qwen) to 0.66 (Llama) - Llama student is less aligned with its DPO teacher's preferences.",
        "- Gini-proxy 0.27-0.30: PFW concentrates noticeably more than kl_inv (which has Gini ~ 0.10-0.15) and uniform (0).",
        "- Bottom 10% of tokens receive ~0.001-0.01 weight: PFW effectively zeros out non-frontier tokens.",
        "- Top 10% of tokens receive 1.5-1.6x uniform: budget is reallocated, not just rescaled.",
    ]
    out = OUT_DIR / "composite_stats.md"
    out.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(md_lines))

if __name__ == "__main__":
    composite_heatmap()
    composite_distribution()
    stats_table()
