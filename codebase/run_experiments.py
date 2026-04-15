"""
Batch experiment runner.
Runs lossless and lossy (multiple quality levels) on all BMP images in data/.
Outputs: results/summary_table.csv + per-image comparison PNGs + metric charts.
"""

import sys
import csv
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bmp_io import read_bmp, BMPReadError
import codec_lossless
import codec_lossy
import metrics as met
import visualize as viz

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOSSY_QUALITIES = [10, 30, 50, 75, 90]

# Image source metadata (for documentation)
IMAGE_SOURCES = {
    "natural":       "Synthetic - procedurally generated natural-like scene (Gaussian blur + noise)",
    "texture":       "Synthetic - repeating sine-wave texture pattern",
    "geometric":     "Synthetic - geometric shapes (circles, rectangles, triangles)",
    "gradient":      "Synthetic - smooth RGB color gradient",
    "low_color":     "Synthetic - 8-color cartoon-like image (horizontal bands)",
    "real_kodak01":  "Kodak Image Dataset kodim01 (flowers) - https://r0k.us/graphics/kodak/",
    "real_kodak23":  "Kodak Image Dataset kodim23 (lighthouse) - https://r0k.us/graphics/kodak/",
    "real_kodak05":  "Kodak Image Dataset kodim05 (toy) - https://r0k.us/graphics/kodak/",
    "real_kodak15":  "Kodak Image Dataset kodim15 (beach) - https://r0k.us/graphics/kodak/",
    "real_sipi_girl":"USC-SIPI Image Database misc/4.1.01 (Girl) - https://sipi.usc.edu/database/",
}


def process_image(bmp_path: Path):
    name = bmp_path.stem
    print(f"\n{'='*60}")
    print(f"Image: {name}  ({IMAGE_SOURCES.get(name, 'unknown source')})")
    print(f"{'='*60}")

    try:
        info = read_bmp(str(bmp_path))
    except BMPReadError as e:
        print(f"  SKIP (BMPReadError): {e}")
        return []

    pixels = info["pixels"]
    print(f"  Size: {info['width']}x{info['height']}  bit_depth={info['bit_depth']}")

    rows = []

    # ── Lossless ──────────────────────────────────────────────────────────────
    hrc_path = str(RESULTS_DIR / f"{name}_lossless.hrc")
    try:
        stats = codec_lossless.compress(pixels, hrc_path)
        pixels_rec, _, dstats = codec_lossless.decompress(hrc_path)

        import numpy as np
        match = np.array_equal(pixels, pixels_rec)
        m = met.compute_all(
            pixels, pixels_rec,
            stats["original_size"], stats["compressed_size"],
            stats["encode_time"], dstats["decode_time"]
        )
        m["lossless_match"] = match
        m["image"] = name
        m["algorithm"] = "lossless_RLE+Huffman"
        m["quality"] = "N/A"
        rows.append(m)

        print(f"  [Lossless] ratio={m['compression_ratio']:.3f}  "
              f"PSNR={m['psnr_db']}  SSIM={m['ssim']}  match={match}")

        viz.save_comparison(
            pixels, pixels_rec,
            str(RESULTS_DIR / f"{name}_lossless_cmp.png"),
            title=f"{name} - Lossless RLE+Huffman",
            metrics=m
        )
    except Exception as e:
        print(f"  [Lossless] ERROR: {e}")

    # ── Lossy (multiple quality levels) ───────────────────────────────────────
    for q in LOSSY_QUALITIES:
        dct_path = str(RESULTS_DIR / f"{name}_lossy_q{q}.dct")
        try:
            stats = codec_lossy.compress(pixels, dct_path, quality=q)
            pixels_rec, dstats = codec_lossy.decompress(dct_path)

            m = met.compute_all(
                pixels, pixels_rec,
                stats["original_size"], stats["compressed_size"],
                stats["encode_time"], dstats["decode_time"]
            )
            m["lossless_match"] = False
            m["image"] = name
            m["algorithm"] = "lossy_DCT"
            m["quality"] = q
            rows.append(m)

            print(f"  [Lossy q={q:3d}] ratio={m['compression_ratio']:.3f}  "
                  f"PSNR={m['psnr_db']}  SSIM={m['ssim']}")

            # Save comparison only for q=50
            if q == 50:
                viz.save_comparison(
                    pixels, pixels_rec,
                    str(RESULTS_DIR / f"{name}_lossy_q50_cmp.png"),
                    title=f"{name} - Lossy DCT (quality=50)",
                    metrics=m
                )
        except Exception as e:
            print(f"  [Lossy q={q}] ERROR: {e}")

    return rows


def main():
    bmp_files = sorted(DATA_DIR.glob("*.bmp"))
    if not bmp_files:
        print("No BMP files found in data/")
        sys.exit(1)

    print(f"Found {len(bmp_files)} BMP files: {[f.name for f in bmp_files]}")

    all_rows = []
    for bmp_path in bmp_files:
        rows = process_image(bmp_path)
        all_rows.extend(rows)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "summary_table.csv"
    if all_rows:
        fieldnames = ["image", "algorithm", "quality",
                      "compression_ratio", "psnr_db", "ssim", "mse",
                      "original_size_bytes", "compressed_size_bytes",
                      "encode_time_s", "decode_time_s", "lossless_match"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved: {csv_path}  ({len(all_rows)} rows)")

    # ── Metric charts ─────────────────────────────────────────────────────────
    images_ordered = [r["image"] for r in all_rows
                      if r["algorithm"] == "lossless_RLE+Huffman"]

    # 1. Compression ratio bar: lossless vs lossy q=50
    ratio_data = {}
    for row in all_rows:
        img = row["image"]
        algo = row["algorithm"]
        q    = row["quality"]
        if algo == "lossy_DCT" and q != 50:
            continue
        label = "Lossless (RLE+Huffman)" if algo == "lossless_RLE+Huffman" else "Lossy DCT q=50"
        ratio_data.setdefault(label, {})[img] = row["compression_ratio"]

    # 2. PSNR multi-quality: lossy only, one line per quality level
    psnr_quality_data = {}
    ssim_quality_data = {}
    for row in all_rows:
        if row["algorithm"] != "lossy_DCT":
            continue
        q = row["quality"]
        label = f"q={q}"
        psnr_quality_data.setdefault(label, {})[row["image"]] = float(row["psnr_db"])
        ssim_quality_data.setdefault(label, {})[row["image"]] = float(row["ssim"])

    # 3. SSIM comparison: lossless vs lossy q=50 (SSIM is meaningful for both)
    ssim_compare_data = {}
    for row in all_rows:
        img = row["image"]
        algo = row["algorithm"]
        q    = row["quality"]
        if algo == "lossy_DCT" and q != 50:
            continue
        label = "Lossless (RLE+Huffman)" if algo == "lossless_RLE+Huffman" else "Lossy DCT q=50"
        ssim_compare_data.setdefault(label, {})[img] = float(row["ssim"])

    if ratio_data:
        viz.save_compression_ratio_bar(
            ratio_data,
            str(RESULTS_DIR / "chart_compression_ratio.png")
        )
        viz.save_psnr_multiquality(
            psnr_quality_data,
            str(RESULTS_DIR / "chart_psnr.png")
        )
        viz.save_metric_linechart(
            ssim_quality_data, "SSIM",
            str(RESULTS_DIR / "chart_ssim.png"),
            ylabel="SSIM"
        )
        print("Charts saved.")

    # ── Metrics table image ────────────────────────────────────────────────────
    table_rows = [r for r in all_rows
                  if r["algorithm"] == "lossless_RLE+Huffman"
                  or (r["algorithm"] == "lossy_DCT" and r["quality"] == 50)]
    display_rows = []
    for r in table_rows:
        display_rows.append({
            "Image": r["image"],
            "Algorithm": r["algorithm"],
            "Ratio": r["compression_ratio"],
            "PSNR(dB)": r["psnr_db"],
            "SSIM": r["ssim"],
            "MSE": r["mse"],
            "EncTime(s)": r["encode_time_s"],
            "DecTime(s)": r["decode_time_s"],
        })
    viz.save_metrics_table(display_rows, str(RESULTS_DIR / "metrics_table.png"))
    print("Metrics table image saved.")

    print(f"\nAll results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
