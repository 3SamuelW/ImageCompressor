"""
Image Compressor CLI
Usage:
  python main.py --image data/natural.bmp --algo lossless
  python main.py --image data/natural.bmp --algo lossy --quality 50
  python main.py --image data/natural.bmp --algo both --quality 50
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bmp_io import read_bmp, write_bmp, BMPReadError
import codec_lossless
import codec_lossy
import metrics as met
import visualize as viz


def run_lossless(pixels, image_name, out_dir):
    hrc_path = str(out_dir / f"{image_name}_lossless.hrc")
    rec_bmp  = str(out_dir / f"{image_name}_lossless_rec.bmp")

    print(f"  [Lossless] Compressing ...")
    stats = codec_lossless.compress(pixels, hrc_path)
    print(f"  [Lossless] Compressed: ratio={stats['compression_ratio']:.3f}  "
          f"time={stats['encode_time']:.3f}s  "
          f"{stats['original_size']}B -> {stats['compressed_size']}B")

    pixels_rec, _, dstats = codec_lossless.decompress(hrc_path)
    print(f"  [Lossless] Decompressed: time={dstats['decode_time']:.3f}s")

    import numpy as np
    match = np.array_equal(pixels, pixels_rec)
    print(f"  [Lossless] Pixel-perfect match: {match}")
    if not match:
        print("  [Lossless] WARNING: lossless verification FAILED")

    write_bmp(rec_bmp, pixels_rec)

    m = met.compute_all(
        pixels, pixels_rec,
        stats["original_size"], stats["compressed_size"],
        stats["encode_time"], dstats["decode_time"]
    )
    print(f"  [Lossless] PSNR={m['psnr_db']} dB  SSIM={m['ssim']}  MSE={m['mse']}")
    return m, pixels_rec


def run_lossy(pixels, image_name, out_dir, quality=50):
    dct_path = str(out_dir / f"{image_name}_lossy_q{quality}.dct")
    rec_bmp  = str(out_dir / f"{image_name}_lossy_q{quality}_rec.bmp")

    print(f"  [Lossy q={quality}] Compressing ...")
    stats = codec_lossy.compress(pixels, dct_path, quality=quality)
    print(f"  [Lossy q={quality}] Compressed: ratio={stats['compression_ratio']:.3f}  "
          f"time={stats['encode_time']:.3f}s  "
          f"{stats['original_size']}B -> {stats['compressed_size']}B")

    pixels_rec, dstats = codec_lossy.decompress(dct_path)
    print(f"  [Lossy q={quality}] Decompressed: time={dstats['decode_time']:.3f}s")

    write_bmp(rec_bmp, pixels_rec)

    m = met.compute_all(
        pixels, pixels_rec,
        stats["original_size"], stats["compressed_size"],
        stats["encode_time"], dstats["decode_time"]
    )
    print(f"  [Lossy q={quality}] PSNR={m['psnr_db']} dB  SSIM={m['ssim']}  MSE={m['mse']}")
    return m, pixels_rec


def main():
    parser = argparse.ArgumentParser(description="BMP Image Compressor")
    parser.add_argument("--image", required=True, help="Path to input BMP file")
    parser.add_argument("--algo", choices=["lossless", "lossy", "both"], default="both")
    parser.add_argument("--quality", type=int, default=50, help="Lossy quality factor 1-100")
    parser.add_argument("--outdir", default=None, help="Output directory (default: results/)")
    args = parser.parse_args()

    # Resolve paths
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = Path(__file__).parent / args.image

    out_dir = Path(args.outdir) if args.outdir else Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.stem

    # Read BMP
    try:
        info = read_bmp(str(image_path))
    except BMPReadError as e:
        print(f"ERROR reading BMP: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    pixels = info["pixels"]
    print(f"Loaded: {image_path.name}  {info['width']}x{info['height']}  {info['bit_depth']}-bit")

    results = {}

    if args.algo in ("lossless", "both"):
        m, rec = run_lossless(pixels, image_name, out_dir)
        results["lossless"] = (m, rec)
        viz.save_comparison(
            pixels, rec,
            str(out_dir / f"{image_name}_lossless_comparison.png"),
            title=f"{image_name} - Lossless (RLE+Huffman)",
            metrics=m
        )

    if args.algo in ("lossy", "both"):
        m, rec = run_lossy(pixels, image_name, out_dir, quality=args.quality)
        results[f"lossy_q{args.quality}"] = (m, rec)
        viz.save_comparison(
            pixels, rec,
            str(out_dir / f"{image_name}_lossy_q{args.quality}_comparison.png"),
            title=f"{image_name} - Lossy DCT (quality={args.quality})",
            metrics=m
        )

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
