"""
Visualization module: publication-quality figures for academic report.

Design principles:
- Perceptually uniform colormaps (viridis/magma) for heatmaps
- Fixed vmax=255 for all difference heatmaps (unified scale)
- Log-scale y-axis for compression ratio (with y=1 reference line)
- Curated color palette, clean spines, 300 DPI output
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from pathlib import Path

# ── Publication style ─────────────────────────────────────────────────────────

# Curated palette: colorblind-safe, high-contrast, publication-ready
# Based on Paul Tol's "Bright" qualitative scheme
_PALETTE = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

_FONT_FAMILY = "DejaVu Sans"

def _apply_style(ax, grid=True, spines=True):
    """Apply clean academic style to an axes."""
    ax.tick_params(labelsize=9, direction="out", length=3, width=0.8)
    if grid:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#CCCCCC")
        ax.set_axisbelow(True)
    if spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)


# ── Comparison figure ─────────────────────────────────────────────────────────

def save_comparison(original: np.ndarray, reconstructed: np.ndarray,
                    out_path: str, title: str = "", metrics: dict = None) -> None:
    """
    3-panel figure: Original | Reconstructed | Difference heatmap.
    - Lossless (zero diff): shows solid green "pixel-perfect" panel
    - Lossy: adaptive vmax = 99th percentile of diff, colormap YlOrRd
    """
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    diff_gray = diff.mean(axis=2) if diff.ndim == 3 else diff
    is_lossless = diff_gray.max() == 0

    fig = plt.figure(figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08,
                           left=0.02, right=0.96, top=0.86, bottom=0.13)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    cmap_img = None if original.ndim == 3 else "gray"
    ax0.imshow(original, cmap=cmap_img, interpolation="nearest")
    ax0.set_title("Original", fontsize=10, fontweight="bold", pad=6)
    ax0.axis("off")

    ax1.imshow(reconstructed, cmap=cmap_img, interpolation="nearest")
    ax1.set_title("Reconstructed", fontsize=10, fontweight="bold", pad=6)
    ax1.axis("off")

    if is_lossless:
        # Solid green panel with text annotation
        ax2.imshow(np.zeros_like(diff_gray), cmap="Greens", vmin=0, vmax=1)
        ax2.text(0.5, 0.5, "Pixel-Perfect\n(diff = 0)",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=13, fontweight="bold", color="#1a7a1a",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#d4f5d4",
                           edgecolor="#1a7a1a", linewidth=1.5))
        ax2.set_title("Difference (|orig − rec|)", fontsize=10, fontweight="bold", pad=6)
        ax2.axis("off")
        # Fake colorbar at 0
        sm = plt.cm.ScalarMappable(cmap="Greens", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.03, shrink=0.92)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Pixel error", fontsize=8, labelpad=4)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "0"])
    else:
        # Adaptive vmax: 99th percentile (avoids outlier saturation)
        vmax = float(np.percentile(diff_gray, 99))
        vmax = max(vmax, 1.0)  # at least 1

        im = ax2.imshow(diff_gray, cmap="YlOrRd", vmin=0, vmax=vmax,
                        interpolation="nearest")
        ax2.set_title("Difference (|orig − rec|)", fontsize=10, fontweight="bold", pad=6)
        ax2.axis("off")

        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03, shrink=0.92)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Pixel error", fontsize=8, labelpad=4)
        # 5 evenly spaced ticks
        ticks = [round(vmax * i / 4) for i in range(5)]
        cbar.set_ticks(ticks)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.95)

    if metrics:
        psnr = metrics.get("psnr_db", "N/A")
        ssim = metrics.get("ssim", "N/A")
        ratio = metrics.get("compression_ratio", "N/A")
        psnr_s = "∞" if str(psnr) == "inf" or psnr == float("inf") else f"{float(psnr):.2f}"
        ssim_s = f"{float(ssim):.4f}" if ssim != "N/A" else "N/A"
        ratio_s = f"{float(ratio):.2f}×" if ratio != "N/A" else "N/A"
        fig.text(0.5, 0.03,
                 f"PSNR = {psnr_s} dB   |   SSIM = {ssim_s}   |   Compression Ratio = {ratio_s}",
                 ha="center", fontsize=9, color="#444444",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                           edgecolor="#CCCCCC", linewidth=0.8))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


# ── Compression ratio chart ───────────────────────────────────────────────────

def save_compression_ratio_bar(data: dict, out_path: str) -> None:
    """
    Grouped bar chart with log-scale y-axis and y=1 reference line.
    Clearly shows which images compress above/below 1×.
    """
    algos = list(data.keys())
    images = list(next(iter(data.values())).keys())

    # Shorten image labels
    short_labels = [img.replace("real_", "").replace("_", "\n") for img in images]

    x = np.arange(len(images))
    n = len(algos)
    total_width = 0.72
    width = total_width / n

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    fig.patch.set_facecolor("white")

    colors = _PALETTE[:n]
    for i, (algo, color) in enumerate(zip(algos, colors)):
        vals = [max(data[algo].get(img, 0), 1e-3) for img in images]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=algo,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.5,
                      zorder=3)

    # y=1 reference line (break-even)
    ax.axhline(y=1.0, color="#CC3333", linewidth=1.4, linestyle="--",
               zorder=4, label="Ratio = 1 (break-even)")

    # Log scale
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}×" if v >= 1 else f"{v:.2f}×"
    ))
    # Custom ticks for readability
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_xlabel("Image", fontsize=10, labelpad=6)
    ax.set_ylabel("Compression Ratio (log scale)", fontsize=10, labelpad=6)
    ax.set_title("Compression Ratio by Algorithm and Image", fontsize=11,
                 fontweight="bold", pad=10)

    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5,
                  alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend = ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC",
                       loc="upper right", ncol=1)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Line charts (PSNR / SSIM) ─────────────────────────────────────────────────

def save_psnr_multiquality(data: dict, out_path: str) -> None:
    """
    PSNR line chart: one line per quality level (lossy only).
    X-axis: images, Y-axis: PSNR (dB).
    Lossless PSNR=inf is annotated as a text band, not plotted.
    """
    # Sort quality levels numerically
    quality_order = sorted(data.keys(), key=lambda s: int(s.split("=")[1]))
    images = list(next(iter(data.values())).keys())
    short = [img.replace("real_", "").replace("_", " ") for img in images]

    # Color gradient from low quality (warm) to high quality (cool)
    q_colors = ["#CC3311", "#EE7733", "#CCBB44", "#228833", "#4477AA"]
    markers  = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    fig.patch.set_facecolor("white")

    for idx, q_label in enumerate(quality_order):
        vals = [data[q_label].get(img, 0) for img in images]
        color = q_colors[idx % len(q_colors)]
        marker = markers[idx % len(markers)]
        ax.plot(short, vals, marker=marker, color=color, label=q_label,
                linewidth=1.6, markersize=6, markeredgewidth=0.8,
                markeredgecolor="white", zorder=3)

    # Annotate lossless PSNR = ∞ as a shaded band at top
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 60
    ax.set_ylim(bottom=15)
    ylim_top = max(v for q in data.values() for v in q.values()) * 1.12
    ax.set_ylim(top=ylim_top)
    ax.axhspan(ylim_top * 0.93, ylim_top, alpha=0.12, color="#4477AA", zorder=0)
    ax.text(len(images) - 1, ylim_top * 0.965,
            "Lossless: PSNR = ∞ (pixel-perfect)",
            ha="right", va="center", fontsize=8, color="#4477AA",
            style="italic")

    ax.set_xlabel("Image", fontsize=10, labelpad=6)
    ax.set_ylabel("PSNR (dB)", fontsize=10, labelpad=6)
    ax.set_title("PSNR vs. Image Content at Different Quality Levels (Lossy DCT)",
                 fontsize=11, fontweight="bold", pad=10)

    _apply_style(ax)
    ax.tick_params(axis="x", rotation=30, labelsize=8.5)

    legend = ax.legend(title="Quality", fontsize=8.5, title_fontsize=9,
                       framealpha=0.9, edgecolor="#CCCCCC", loc="lower right")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def save_metric_linechart(data: dict, metric: str, out_path: str,
                          xlabel: str = "Image", ylabel: str = None) -> None:
    """
    Publication-quality line chart with markers.
    """
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    fig.patch.set_facecolor("white")

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors = _PALETTE

    for idx, (algo, img_vals) in enumerate(data.items()):
        images = list(img_vals.keys())
        short = [img.replace("real_", "").replace("_", " ") for img in images]
        values = [img_vals[img] for img in images]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(short, values, marker=marker, color=color, label=algo,
                linewidth=1.6, markersize=6, markeredgewidth=0.8,
                markeredgecolor="white", zorder=3)

    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel or metric, fontsize=10, labelpad=6)
    ax.set_title(f"{metric} Across Images", fontsize=11, fontweight="bold", pad=10)

    _apply_style(ax)
    ax.tick_params(axis="x", rotation=30, labelsize=8.5)

    legend = ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC",
                       loc="best", ncol=1)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Metrics table ─────────────────────────────────────────────────────────────

def save_metrics_table(rows: list, out_path: str) -> None:
    """Styled metrics table image."""
    if not rows:
        return
    keys = list(rows[0].keys())
    cell_text = [[str(row.get(k, "")) for k in keys] for row in rows]

    fig, ax = plt.subplots(figsize=(max(14, len(keys) * 1.8),
                                    max(3, len(rows) * 0.45 + 1.2)), dpi=150)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=keys,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(keys))))

    # Style header
    for j in range(len(keys)):
        cell = table[0, j]
        cell.set_facecolor("#4477AA")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(keys)):
            cell = table[i, j]
            cell.set_facecolor("#F0F4FA" if i % 2 == 0 else "white")
            cell.set_edgecolor("#DDDDDD")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    rec = np.clip(orig.astype(int) + rng.integers(-20, 20, orig.shape), 0, 255).astype(np.uint8)
    save_comparison(orig, rec, "test_comparison.png", title="Test",
                    metrics={"psnr_db": 30.0, "ssim": 0.95, "compression_ratio": 2.5})
    print("Saved test_comparison.png")
