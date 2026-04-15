"""
Lossless codec: PNG-style prediction filter + RLE + Huffman entropy coding.
Output format: custom .hrc binary file.

Pipeline:
  Encode: pixels -> prediction filter (per row) -> residuals -> RLE -> Huffman -> .hrc
  Decode: .hrc -> Huffman -> RLE -> residuals -> inverse filter -> pixels

Prediction filters (PNG spec):
  0 = None    : residual = pixel
  1 = Sub     : residual = pixel - left
  2 = Up      : residual = pixel - above
  3 = Average : residual = pixel - floor((left + above) / 2)
  4 = Paeth   : residual = pixel - paeth_predictor(left, above, upper-left)

Best filter is chosen per row to minimize sum of absolute residuals.

.hrc file layout:
  [4B magic "HRC2"]
  [4B width (uint32)]
  [4B height (uint32)]
  [1B bit_depth (uint8)]
  [4B palette_entries (uint32, 0 for 24-bit)]
  [palette_entries * 3 bytes RGB palette]
  [filter_map: height bytes, one filter id per row]
  [4B symbol_count (uint32)]
  [4B bitstream_len_bits (uint32)]
  [codebook: see _ser_codebook]
  [bitstream bytes]
"""

import struct
import heapq
import numpy as np
from pathlib import Path
from collections import Counter
import time


# ── PNG prediction filters ────────────────────────────────────────────────────

def _paeth(a, b, c):
    """Paeth predictor (PNG spec). Works on int arrays."""
    p = a.astype(np.int32) + b.astype(np.int32) - c.astype(np.int32)
    pa = np.abs(p - a.astype(np.int32))
    pb = np.abs(p - b.astype(np.int32))
    pc = np.abs(p - c.astype(np.int32))
    result = np.where(pa <= pb, np.where(pa <= pc, a, c), np.where(pb <= pc, b, c))
    return result.astype(np.uint8)


def _apply_filter(row: np.ndarray, prev_row: np.ndarray, ftype: int) -> np.ndarray:
    """Apply PNG filter to a row. row and prev_row are uint8 1D arrays (bytes)."""
    row = row.astype(np.int32)
    prev = prev_row.astype(np.int32)
    if ftype == 0:
        return (row % 256).astype(np.uint8)
    elif ftype == 1:  # Sub
        left = np.roll(row, 1); left[0] = 0
        return ((row - left) % 256).astype(np.uint8)
    elif ftype == 2:  # Up
        return ((row - prev) % 256).astype(np.uint8)
    elif ftype == 3:  # Average
        left = np.roll(row, 1); left[0] = 0
        avg = (left + prev) // 2
        return ((row - avg) % 256).astype(np.uint8)
    elif ftype == 4:  # Paeth
        left = np.roll(row.astype(np.uint8), 1); left[0] = 0
        ul = np.roll(prev_row, 1); ul[0] = 0
        pred = _paeth(left, prev_row, ul).astype(np.int32)
        return ((row - pred) % 256).astype(np.uint8)
    else:
        raise ValueError(f"Unknown filter type {ftype}")


def _reverse_filter(residual: np.ndarray, prev_row: np.ndarray, ftype: int) -> np.ndarray:
    """Reverse PNG filter to recover original row."""
    res = residual.astype(np.int32)
    prev = prev_row.astype(np.int32)
    if ftype == 0:
        return residual.copy()
    elif ftype == 1:  # Sub
        out = np.zeros_like(res)
        for i in range(len(res)):
            left = out[i-1] if i > 0 else 0
            out[i] = (res[i] + left) % 256
        return out.astype(np.uint8)
    elif ftype == 2:  # Up
        return ((res + prev) % 256).astype(np.uint8)
    elif ftype == 3:  # Average
        out = np.zeros_like(res)
        for i in range(len(res)):
            left = out[i-1] if i > 0 else 0
            avg = (left + prev[i]) // 2
            out[i] = (res[i] + avg) % 256
        return out.astype(np.uint8)
    elif ftype == 4:  # Paeth
        out = np.zeros(len(res), dtype=np.uint8)
        for i in range(len(res)):
            left = int(out[i-1]) if i > 0 else 0
            above = int(prev[i])
            ul = int(prev[i-1]) if i > 0 else 0
            p = left + above - ul
            pa = abs(p - left); pb = abs(p - above); pc = abs(p - ul)
            if pa <= pb and pa <= pc:
                pred = left
            elif pb <= pc:
                pred = above
            else:
                pred = ul
            out[i] = (res[i] + pred) % 256
        return out
    else:
        raise ValueError(f"Unknown filter type {ftype}")


def _best_filter(row: np.ndarray, prev_row: np.ndarray) -> int:
    """Choose filter type that minimizes sum of absolute residuals."""
    best_f, best_score = 0, float('inf')
    for f in range(5):
        res = _apply_filter(row, prev_row, f).astype(np.int8)
        score = np.sum(np.abs(res.astype(np.int32)))
        if score < best_score:
            best_score = score; best_f = f
    return best_f


def _filter_image(raw_bytes: bytes, width: int, height: int, bpp: int):
    """
    Apply per-row best PNG filter.
    bpp: bytes per pixel (3 for 24-bit, 1 for 8-bit).
    Returns (filtered_bytes, filter_map).
    """
    row_bytes = width * bpp
    filtered = bytearray()
    filter_map = []
    prev_row = np.zeros(row_bytes, dtype=np.uint8)
    for y in range(height):
        row = np.frombuffer(raw_bytes[y*row_bytes:(y+1)*row_bytes], dtype=np.uint8)
        f = _best_filter(row, prev_row)
        res = _apply_filter(row, prev_row, f)
        filtered.extend(res.tobytes())
        filter_map.append(f)
        prev_row = row
    return bytes(filtered), filter_map


def _unfilter_image(filtered_bytes: bytes, width: int, height: int, bpp: int, filter_map: list):
    """Reverse per-row PNG filter."""
    row_bytes = width * bpp
    out = bytearray()
    prev_row = np.zeros(row_bytes, dtype=np.uint8)
    for y in range(height):
        res = np.frombuffer(filtered_bytes[y*row_bytes:(y+1)*row_bytes], dtype=np.uint8)
        row = _reverse_filter(res, prev_row, filter_map[y])
        out.extend(row.tobytes())
        prev_row = row
    return bytes(out)


# ── RLE ───────────────────────────────────────────────────────────────────────

def rle_encode(data: bytes) -> list:
    if not data: return []
    runs = []
    i = 0; n = len(data)
    while i < n:
        val = data[i]; count = 1
        while i + count < n and data[i+count] == val and count < 255:
            count += 1
        runs.append((val, count)); i += count
    return runs


def rle_decode(runs: list) -> bytes:
    out = bytearray()
    for val, count in runs:
        out += bytes([val]) * count
    return bytes(out)


def runs_to_symbols(runs: list) -> list:
    syms = []
    for val, count in runs:
        syms.append(val); syms.append(count)
    return syms


def symbols_to_runs(symbols: list) -> list:
    return [(symbols[i], symbols[i+1]) for i in range(0, len(symbols), 2)]


# ── Huffman ───────────────────────────────────────────────────────────────────

class HuffmanNode:
    def __init__(self, symbol, freq, left=None, right=None):
        self.symbol = symbol; self.freq = freq
        self.left = left; self.right = right
    def __lt__(self, other): return self.freq < other.freq


def build_huffman_tree(freq: dict) -> HuffmanNode:
    heap = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        node = heapq.heappop(heap)
        return HuffmanNode(None, node.freq, left=node)
    while len(heap) > 1:
        a = heapq.heappop(heap); b = heapq.heappop(heap)
        merged = HuffmanNode(None, a.freq + b.freq, left=a, right=b)
        heapq.heappush(heap, merged)
    return heap[0]


def build_codebook(root: HuffmanNode) -> dict:
    codebook = {}
    def traverse(node, code):
        if node is None: return
        if node.symbol is not None:
            codebook[node.symbol] = code if code else "0"; return
        traverse(node.left, code + "0"); traverse(node.right, code + "1")
    traverse(root, "")
    return codebook


def huffman_encode(symbols: list) -> tuple:
    freq = Counter(symbols)
    root = build_huffman_tree(freq)
    codebook = build_codebook(root)
    bitstring = "".join(codebook[s] for s in symbols)
    return codebook, bitstring


def bitstring_to_bytes(bitstring: str) -> bytes:
    pad = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * pad
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i+8], 2))
    return bytes(out)


def bytes_to_bitstring(data: bytes, total_bits: int) -> str:
    bits = "".join(f"{b:08b}" for b in data)
    return bits[:total_bits]


def huffman_decode(codebook: dict, bitstring: str, n_symbols: int) -> list:
    reverse = {v: k for k, v in codebook.items()}
    symbols = []; buf = ""
    for bit in bitstring:
        buf += bit
        if buf in reverse:
            symbols.append(reverse[buf]); buf = ""
            if len(symbols) == n_symbols: break
    return symbols


def serialise_codebook(codebook: dict) -> bytes:
    out = struct.pack("<I", len(codebook))
    for sym, code in codebook.items():
        code_len = len(code)
        code_bytes = bitstring_to_bytes(code)
        out += struct.pack("<HI", sym, code_len) + code_bytes
    return out


def deserialise_codebook(data: bytes, offset: int) -> tuple:
    count = struct.unpack_from("<I", data, offset)[0]; offset += 4
    codebook = {}
    for _ in range(count):
        sym, code_len = struct.unpack_from("<HI", data, offset); offset += 6
        n_bytes = (code_len + 7) // 8
        code_bytes = data[offset:offset+n_bytes]; offset += n_bytes
        codebook[sym] = bytes_to_bitstring(code_bytes, code_len)
    return codebook, offset


# ── High-level compress / decompress ─────────────────────────────────────────

def compress(pixels: np.ndarray, out_path: str, palette=None) -> dict:
    t0 = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pixels.ndim == 3:
        bit_depth = 24
        height, width = pixels.shape[:2]
        bpp = 3
        raw_bytes = pixels.tobytes()
    else:
        bit_depth = 8
        height, width = pixels.shape
        bpp = 1
        raw_bytes = pixels.tobytes()

    # PNG prediction filter
    filtered_bytes, filter_map = _filter_image(raw_bytes, width, height, bpp)

    # RLE encode
    runs = rle_encode(filtered_bytes)
    symbols = runs_to_symbols(runs)
    n_symbols = len(symbols)

    # Huffman encode
    codebook, bitstring = huffman_encode(symbols)
    compressed_bits = len(bitstring)
    compressed_bytes = bitstring_to_bytes(bitstring)

    # Build file
    out = bytearray()
    out += b"HRC2"
    out += struct.pack("<III", width, height, bit_depth)

    # Palette
    if bit_depth == 8 and palette:
        out += struct.pack("<I", 256)
        for r, g, b in palette:
            out += bytes([r, g, b])
    else:
        out += struct.pack("<I", 0)

    # Filter map (one byte per row)
    out += bytes(filter_map)

    # Codebook + stream
    out += serialise_codebook(codebook)
    out += struct.pack("<I", n_symbols)
    out += struct.pack("<I", compressed_bits)
    out += compressed_bytes

    with open(out_path, "wb") as f:
        f.write(out)

    elapsed = time.time() - t0
    original_size = len(raw_bytes)
    compressed_size = len(out)
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": original_size / compressed_size,
        "encode_time": elapsed,
    }


def decompress(in_path: str) -> tuple:
    t0 = time.time()
    with open(in_path, "rb") as f:
        data = f.read()

    magic = data[:4]
    if magic not in (b"HRC1", b"HRC2"):
        raise ValueError(f"Not a valid .hrc file: {in_path}")

    offset = 4
    width, height, bit_depth = struct.unpack_from("<III", data, offset); offset += 12
    bpp = 3 if bit_depth == 24 else 1

    palette_entries = struct.unpack_from("<I", data, offset)[0]; offset += 4
    palette = None
    if palette_entries > 0:
        palette = []
        for _ in range(palette_entries):
            r, g, b = data[offset], data[offset+1], data[offset+2]
            palette.append((r, g, b)); offset += 3

    if magic == b"HRC2":
        # Read filter map
        filter_map = list(data[offset:offset+height]); offset += height
    else:
        filter_map = None  # legacy: no filter

    codebook, offset = deserialise_codebook(data, offset)
    n_symbols = struct.unpack_from("<I", data, offset)[0]; offset += 4
    compressed_bits = struct.unpack_from("<I", data, offset)[0]; offset += 4
    compressed_bytes_data = data[offset:]
    bitstring = bytes_to_bitstring(compressed_bytes_data, compressed_bits)

    symbols = huffman_decode(codebook, bitstring, n_symbols)
    runs = symbols_to_runs(symbols)
    filtered_bytes = rle_decode(runs)

    if magic == b"HRC2":
        raw_bytes = _unfilter_image(filtered_bytes, width, height, bpp, filter_map)
    else:
        raw_bytes = filtered_bytes

    if bit_depth == 24:
        pixels = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, 3)
    else:
        pixels = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width)

    elapsed = time.time() - t0
    return pixels, palette, {"decode_time": elapsed}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from bmp_io import read_bmp, write_bmp
    import hashlib

    if len(sys.argv) < 2:
        print("Usage: python codec_lossless.py <file.bmp>")
        sys.exit(1)

    src = sys.argv[1]
    info = read_bmp(src)
    hrc_path = src.replace(".bmp", ".hrc")

    stats = compress(info["pixels"], hrc_path, info["palette"])
    print(f"Compressed: ratio={stats['compression_ratio']:.3f}  time={stats['encode_time']:.3f}s")

    pixels_rec, palette_rec, dstats = decompress(hrc_path)
    print(f"Decompressed: time={dstats['decode_time']:.3f}s")

    match = np.array_equal(info["pixels"], pixels_rec)
    print(f"Pixel-level lossless match: {match}")
    assert match, "LOSSLESS VERIFICATION FAILED"
