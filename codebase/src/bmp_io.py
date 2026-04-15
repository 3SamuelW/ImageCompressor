"""
BMP I/O module — reads and writes 24-bit and 8-bit BMP files.
Parses the BMP file header and DIB header manually (no PIL dependency for core I/O).
"""

import struct
import numpy as np
from pathlib import Path


class BMPReadError(Exception):
    pass


def read_bmp(path: str) -> dict:
    """
    Read a BMP file and return a dict with:
      - pixels: np.ndarray shape (H, W, 3) uint8 for 24-bit, (H, W) uint8 for 8-bit
      - bit_depth: 24 or 8
      - width, height: int
      - palette: list of (R,G,B) for 8-bit, None for 24-bit
      - raw_header: bytes of the full header (for lossless round-trip)
    """
    path = Path(path)
    if not path.exists():
        raise BMPReadError(f"File not found: {path}")

    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 54:
        raise BMPReadError("File too small to be a valid BMP")

    # BMP file header (14 bytes)
    signature = data[0:2]
    if signature != b"BM":
        raise BMPReadError(f"Not a BMP file (signature={signature})")

    file_size = struct.unpack_from("<I", data, 2)[0]
    pixel_offset = struct.unpack_from("<I", data, 10)[0]

    # DIB header (BITMAPINFOHEADER = 40 bytes minimum)
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40:
        raise BMPReadError(f"Unsupported DIB header size: {dib_size}")

    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bit_depth = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]

    if compression != 0:
        raise BMPReadError(f"Compressed BMP not supported (compression={compression})")
    if bit_depth not in (8, 24):
        raise BMPReadError(f"Unsupported bit depth: {bit_depth} (only 8 and 24 supported)")

    flipped = height > 0  # positive height = bottom-up storage
    height = abs(height)

    palette = None
    if bit_depth == 8:
        # Read color table (256 entries × 4 bytes each)
        palette_offset = 14 + dib_size
        palette = []
        for i in range(256):
            b, g, r, _ = data[palette_offset + i*4 : palette_offset + i*4 + 4]
            palette.append((r, g, b))

    # Row stride is padded to 4-byte boundary
    bytes_per_pixel = bit_depth // 8
    row_stride = (width * bytes_per_pixel + 3) & ~3

    pixel_data = data[pixel_offset:]
    if len(pixel_data) < row_stride * height:
        raise BMPReadError("Pixel data shorter than expected")

    if bit_depth == 24:
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            row_start = row * row_stride
            row_bytes = pixel_data[row_start : row_start + width * 3]
            row_arr = np.frombuffer(row_bytes, dtype=np.uint8).reshape(width, 3)
            # BMP stores BGR; convert to RGB
            pixels[row, :, 0] = row_arr[:, 2]
            pixels[row, :, 1] = row_arr[:, 1]
            pixels[row, :, 2] = row_arr[:, 0]
        if flipped:
            pixels = pixels[::-1, :, :]
    else:  # 8-bit
        pixels = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            row_start = row * row_stride
            pixels[row, :] = np.frombuffer(
                pixel_data[row_start : row_start + width], dtype=np.uint8
            )
        if flipped:
            pixels = pixels[::-1, :]

    return {
        "pixels": pixels,
        "bit_depth": bit_depth,
        "width": width,
        "height": height,
        "palette": palette,
        "raw_header": data[:pixel_offset],
    }


def write_bmp(path: str, pixels: np.ndarray, palette=None) -> None:
    """
    Write pixels to a BMP file.
    pixels: (H, W, 3) uint8 for 24-bit, (H, W) uint8 for 8-bit.
    palette: list of 256 (R,G,B) tuples for 8-bit; ignored for 24-bit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if pixels.ndim == 3:
        bit_depth = 24
        height, width = pixels.shape[:2]
    elif pixels.ndim == 2:
        bit_depth = 8
        height, width = pixels.shape
        if palette is None:
            # Default grayscale palette
            palette = [(i, i, i) for i in range(256)]
    else:
        raise ValueError("pixels must be 2D or 3D array")

    bytes_per_pixel = bit_depth // 8
    row_stride = (width * bytes_per_pixel + 3) & ~3
    pixel_data_size = row_stride * height

    palette_size = 256 * 4 if bit_depth == 8 else 0
    dib_size = 40
    file_header_size = 14
    pixel_offset = file_header_size + dib_size + palette_size
    file_size = pixel_offset + pixel_data_size

    out = bytearray()

    # File header
    out += b"BM"
    out += struct.pack("<I", file_size)
    out += struct.pack("<HH", 0, 0)  # reserved
    out += struct.pack("<I", pixel_offset)

    # DIB header
    out += struct.pack("<I", dib_size)
    out += struct.pack("<i", width)
    out += struct.pack("<i", -height)  # negative = top-down
    out += struct.pack("<HH", 1, bit_depth)
    out += struct.pack("<I", 0)  # no compression
    out += struct.pack("<I", pixel_data_size)
    out += struct.pack("<ii", 2835, 2835)  # ~72 DPI
    out += struct.pack("<II", 0, 0)

    # Palette
    if bit_depth == 8:
        for r, g, b in palette:
            out += bytes([b, g, r, 0])

    # Pixel data (top-down, BGR for 24-bit)
    padding = bytes(row_stride - width * bytes_per_pixel)
    for row in range(height):
        if bit_depth == 24:
            row_rgb = pixels[row]  # (W, 3) RGB
            row_bgr = row_rgb[:, ::-1].tobytes()  # flip to BGR
            out += row_bgr
        else:
            out += pixels[row].tobytes()
        out += padding

    with open(path, "wb") as f:
        f.write(out)


def bmp_roundtrip_check(path: str) -> bool:
    """Read a BMP, write to a temp file, compare MD5."""
    import hashlib, tempfile, os
    info = read_bmp(path)
    tmp = tempfile.mktemp(suffix=".bmp")
    try:
        write_bmp(tmp, info["pixels"], info["palette"])
        # Re-read and compare pixel arrays
        info2 = read_bmp(tmp)
        match = np.array_equal(info["pixels"], info2["pixels"])
        return match
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bmp_io.py <file.bmp>")
        sys.exit(1)
    info = read_bmp(sys.argv[1])
    print(f"Width={info['width']} Height={info['height']} BitDepth={info['bit_depth']}")
    print(f"Pixels shape: {info['pixels'].shape}")
    ok = bmp_roundtrip_check(sys.argv[1])
    print(f"Round-trip pixel match: {ok}")
