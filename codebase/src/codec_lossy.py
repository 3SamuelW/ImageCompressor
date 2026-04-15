"""
Lossy codec: 8x8 block DCT + JPEG-standard quantization + full entropy coding.
Complete pipeline: YCbCr -> DCT -> Quantize -> Zigzag -> DC-diff + AC-RLE -> Huffman

Key design (correct JPEG approach):
  - Huffman codes ONLY encode: DC category (0-11) and AC (run,cat) symbols (0-255)
  - Amplitude bits are written DIRECTLY into the bitstream (not Huffman-coded)
  - This avoids symbol namespace collisions

.dct file layout:
  [4B magic "DCT2"]
  [4B width uint32][4B height uint32][1B channels][4B quality]
  [4B padded_width][4B padded_height][4B reserved]
  [per-channel: 4B blob_len, then blob]
  blob = [DC Huffman codebook][AC Huffman codebook][4B n_bits][bitstream]
"""

import struct
import heapq
import numpy as np
from pathlib import Path
from collections import Counter
import time


# ── JPEG standard quantization tables ────────────────────────────────────────

JPEG_LUMA_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

JPEG_CHROMA_Q50 = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)

# Zigzag: maps zigzag-position i -> row-major index in 8x8 block
ZIGZAG_POS_TO_IDX = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]


def quality_to_qtable(quality: int, base_table: np.ndarray) -> np.ndarray:
    quality = max(1, min(100, quality))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.floor((base_table * scale + 50) / 100).clip(1, 255).astype(np.float32)


def dct2d(block):
    from scipy.fft import dctn
    return dctn(block, norm="ortho")


def idct2d(block):
    from scipy.fft import idctn
    return idctn(block, norm="ortho")


def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    Y  =  0.299   * img[:,:,0] + 0.587   * img[:,:,1] + 0.114   * img[:,:,2]
    Cb = -0.16874 * img[:,:,0] - 0.33126 * img[:,:,1] + 0.5     * img[:,:,2] + 128
    Cr =  0.5     * img[:,:,0] - 0.41869 * img[:,:,1] - 0.08131 * img[:,:,2] + 128
    return np.stack([Y, Cb, Cr], axis=2)


def ycbcr_to_rgb(img):
    Y = img[:,:,0]; Cb = img[:,:,1] - 128; Cr = img[:,:,2] - 128
    R = Y + 1.40200 * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.77200 * Cb
    return np.clip(np.stack([R, G, B], axis=2), 0, 255).astype(np.uint8)


def pad_to_multiple(channel, block=8):
    h, w = channel.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    return np.pad(channel, ((0, ph), (0, pw)), mode="edge")


# ── VLI (Variable Length Integer) helpers ────────────────────────────────────

def vli_category(v: int) -> int:
    """Number of bits needed to represent |v|. 0 for v=0."""
    if v == 0: return 0
    v = abs(v)
    cat = 0
    while v > 0:
        v >>= 1; cat += 1
    return cat


def vli_encode(v: int, cat: int) -> str:
    """Return 'cat' bits representing v in JPEG VLI format."""
    if cat == 0: return ""
    if v > 0:
        return format(v, f'0{cat}b')
    else:
        # Negative: encode as (v + 2^cat - 1)
        return format(v + (1 << cat) - 1, f'0{cat}b')


def vli_decode(bits: str, cat: int) -> int:
    """Decode 'cat' bits from JPEG VLI format back to signed int."""
    if cat == 0: return 0
    amp = int(bits, 2)
    if amp >= (1 << (cat - 1)):
        return amp
    return amp - (1 << cat) + 1


# ── Huffman ───────────────────────────────────────────────────────────────────

class _HNode:
    __slots__ = ("sym", "freq", "left", "right")
    def __init__(self, sym, freq, left=None, right=None):
        self.sym = sym; self.freq = freq; self.left = left; self.right = right
    def __lt__(self, o): return self.freq < o.freq


def _build_huffman(freq: dict) -> dict:
    """Build Huffman codebook {symbol: bitstring} from frequency dict."""
    heap = [_HNode(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        n = heapq.heappop(heap)
        root = _HNode(None, n.freq, left=n)
    else:
        while len(heap) > 1:
            a = heapq.heappop(heap); b = heapq.heappop(heap)
            heapq.heappush(heap, _HNode(None, a.freq + b.freq, a, b))
        root = heap[0]
    cb = {}
    def walk(node, code):
        if node is None: return
        if node.sym is not None: cb[node.sym] = code or "0"; return
        walk(node.left, code + "0"); walk(node.right, code + "1")
    walk(root, "")
    return cb


def _ser_codebook(cb: dict) -> bytes:
    """Serialize codebook: [4B count] [per entry: 2B sym, 4B code_len, ceil(len/8) bytes]"""
    out = struct.pack("<I", len(cb))
    for sym, code in cb.items():
        nb = (len(code) + 7) // 8
        pad = nb * 8 - len(code)
        out += struct.pack("<HI", sym, len(code))
        out += bytes(int(code[i:i+8].ljust(8,'0'), 2) for i in range(0, nb*8, 8))
    return out


def _deser_codebook(data: bytes, offset: int):
    count = struct.unpack_from("<I", data, offset)[0]; offset += 4
    cb = {}
    for _ in range(count):
        sym, code_len = struct.unpack_from("<HI", data, offset); offset += 6
        nb = (code_len + 7) // 8
        raw = data[offset:offset+nb]; offset += nb
        bits = "".join(f"{b:08b}" for b in raw)[:code_len]
        cb[sym] = bits
    return cb, offset


# ── Channel encode/decode ─────────────────────────────────────────────────────

def _encode_channel(channel: np.ndarray, qtable: np.ndarray) -> bytes:
    """
    Encode one channel to bytes.
    Huffman codes: DC category (0-11), AC (run<<4|cat) symbols
    Amplitude bits: written raw (VLI) directly into bitstream
    """
    h, w = channel.shape
    assert h % 8 == 0 and w % 8 == 0

    # Pass 1: DCT+quantize all blocks, store zigzag sequences
    all_zz = []
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            block = channel[r:r+8, c:c+8].astype(np.float32) - 128.0
            quant = np.round(dct2d(block) / qtable).astype(np.int32).flatten()
            all_zz.append([quant[ZIGZAG_POS_TO_IDX[i]] for i in range(64)])

    # Pass 2: collect Huffman symbol frequencies
    dc_freq = Counter()
    ac_freq = Counter()
    prev_dc = 0
    for zz in all_zz:
        dc_freq[vli_category(zz[0] - prev_dc)] += 1
        prev_dc = zz[0]
        run = 0
        last_nonzero = max((k for k in range(1, 64) if zz[k] != 0), default=0)
        for k in range(1, 64):
            if zz[k] == 0:
                if k > last_nonzero:
                    break  # trailing zeros -> EOB, no ZRL
                run += 1
                if run == 16:
                    ac_freq[0xF0] += 1
                    run = 0
            else:
                ac_freq[(run << 4) | vli_category(zz[k])] += 1
                run = 0
        ac_freq[0x00] += 1  # EOB

    # Build Huffman tables
    dc_cb = _build_huffman(dc_freq)
    ac_cb = _build_huffman(ac_freq)

    # Pass 3: encode bitstream
    bits = []
    prev_dc = 0
    for zz in all_zz:
        # DC: Huffman(category) + VLI(dc_diff)
        dc_diff = zz[0] - prev_dc
        prev_dc = zz[0]
        cat = vli_category(dc_diff)
        bits.append(dc_cb[cat])
        if cat > 0:
            bits.append(vli_encode(dc_diff, cat))

        # AC: Huffman(run<<4|cat) + VLI(ac_value)
        run = 0
        last_nonzero = max((k for k in range(1, 64) if zz[k] != 0), default=0)
        for k in range(1, 64):
            ac = zz[k]
            if ac == 0:
                if k > last_nonzero:
                    break  # trailing zeros -> EOB
                run += 1
                if run == 16:
                    bits.append(ac_cb[0xF0])
                    run = 0
            else:
                cat = vli_category(ac)
                bits.append(ac_cb[(run << 4) | cat])
                bits.append(vli_encode(ac, cat))
                run = 0
        bits.append(ac_cb[0x00])  # EOB

    bitstream = "".join(bits)
    n_bits = len(bitstream)
    pad = (8 - n_bits % 8) % 8
    bitstream += "0" * pad
    bs_bytes = bytes(int(bitstream[i:i+8], 2) for i in range(0, len(bitstream), 8))

    return _ser_codebook(dc_cb) + _ser_codebook(ac_cb) + struct.pack("<I", n_bits) + bs_bytes


def _decode_channel(blob: bytes, padded_h: int, padded_w: int,
                    qtable: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Decode one channel from blob."""
    offset = 0
    dc_cb, offset = _deser_codebook(blob, offset)
    ac_cb, offset = _deser_codebook(blob, offset)
    n_bits = struct.unpack_from("<I", blob, offset)[0]; offset += 4
    nb = (n_bits + 7) // 8
    bits = "".join(f"{b:08b}" for b in blob[offset:offset+nb])[:n_bits]

    # Build reverse codebooks
    dc_rev = {v: k for k, v in dc_cb.items()}
    ac_rev = {v: k for k, v in ac_cb.items()}

    channel = np.zeros((padded_h, padded_w), dtype=np.float32)
    prev_dc = 0
    bi = 0  # bit index

    def read_huffman(rev):
        nonlocal bi
        buf = ""
        while bi < len(bits):
            buf += bits[bi]; bi += 1
            if buf in rev: return rev[buf]
        raise ValueError("Huffman decode overrun")

    def read_bits(n):
        nonlocal bi
        s = bits[bi:bi+n]; bi += n
        return s

    for r in range(0, padded_h, 8):
        for c in range(0, padded_w, 8):
            zz = [0] * 64

            # DC
            cat = read_huffman(dc_rev)
            dc_diff = vli_decode(read_bits(cat), cat)
            dc = prev_dc + dc_diff
            prev_dc = dc
            zz[0] = dc

            # AC
            k = 1
            while k < 64:
                rs = read_huffman(ac_rev)
                if rs == 0x00: break       # EOB
                if rs == 0xF0: k += 16; continue  # ZRL
                run = (rs >> 4) & 0xF
                cat = rs & 0xF
                ac = vli_decode(read_bits(cat), cat)
                k += run
                if k < 64: zz[k] = ac
                k += 1
            else:
                # k reached 64 without EOB: consume the EOB symbol
                read_huffman(ac_rev)

            # Inverse zigzag
            flat = [0] * 64
            for i in range(64):
                flat[ZIGZAG_POS_TO_IDX[i]] = zz[i]
            quant = np.array(flat, dtype=np.float32).reshape(8, 8)
            channel[r:r+8, c:c+8] = idct2d(quant * qtable) + 128.0

    return channel[:orig_h, :orig_w]


# ── Public API ────────────────────────────────────────────────────────────────

def compress(pixels: np.ndarray, out_path: str, quality: int = 50) -> dict:
    t0 = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pixels.ndim == 3:
        channels_data = rgb_to_ycbcr(pixels)
        n_channels = 3; height, width = pixels.shape[:2]
    else:
        channels_data = pixels.astype(np.float32)[:, :, np.newaxis]
        n_channels = 1; height, width = pixels.shape

    luma_q = quality_to_qtable(quality, JPEG_LUMA_Q50)
    chroma_q = quality_to_qtable(quality, JPEG_CHROMA_Q50)

    padded_h = padded_w = None
    blobs = []
    for ch in range(n_channels):
        chan = channels_data[:, :, ch]
        padded = pad_to_multiple(chan)
        if padded_h is None: padded_h, padded_w = padded.shape
        qtable = luma_q if ch == 0 else chroma_q
        blobs.append(_encode_channel(padded, qtable))

    out = bytearray()
    out += b"DCT2"
    out += struct.pack("<IIBIIII", width, height, n_channels, quality, padded_w, padded_h, 0)
    for blob in blobs:
        out += struct.pack("<I", len(blob)) + blob

    with open(out_path, "wb") as f:
        f.write(out)

    elapsed = time.time() - t0
    original_size = pixels.nbytes
    compressed_size = len(out)
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": original_size / compressed_size,
        "encode_time": elapsed,
        "quality": quality,
    }


def decompress(in_path: str) -> tuple:
    t0 = time.time()
    with open(in_path, "rb") as f:
        data = f.read()

    magic = data[:4]
    if magic not in (b"DCT2", b"DCT1"):
        raise ValueError(f"Not a valid .dct file: {in_path}")

    offset = 4
    width, height, n_channels, quality, padded_w, padded_h, _ = struct.unpack_from(
        "<IIBIIII", data, offset)
    offset += struct.calcsize("<IIBIIII")

    luma_q = quality_to_qtable(quality, JPEG_LUMA_Q50)
    chroma_q = quality_to_qtable(quality, JPEG_CHROMA_Q50)

    channels_out = []

    if magic == b"DCT1":
        # Legacy: raw int16 coefficients
        coeff_size = padded_h * padded_w * 2
        for ch in range(n_channels):
            coeffs = np.frombuffer(data[offset:offset+coeff_size], dtype=np.int16).reshape(padded_h, padded_w)
            offset += coeff_size
            qtable = luma_q if ch == 0 else chroma_q
            chan = np.zeros((padded_h, padded_w), dtype=np.float32)
            for r in range(0, padded_h, 8):
                for c in range(0, padded_w, 8):
                    chan[r:r+8, c:c+8] = idct2d(coeffs[r:r+8, c:c+8].astype(np.float32) * qtable) + 128.0
            channels_out.append(chan[:height, :width])
    else:
        for ch in range(n_channels):
            blob_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
            blob = data[offset:offset+blob_len]; offset += blob_len
            qtable = luma_q if ch == 0 else chroma_q
            channels_out.append(_decode_channel(blob, padded_h, padded_w, qtable, height, width))

    if n_channels == 3:
        pixels = ycbcr_to_rgb(np.stack(channels_out, axis=2))
    else:
        pixels = np.clip(channels_out[0], 0, 255).astype(np.uint8)

    elapsed = time.time() - t0
    return pixels, {"decode_time": elapsed, "quality": quality}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from bmp_io import read_bmp

    if len(sys.argv) < 2:
        print("Usage: python codec_lossy.py <file.bmp> [quality=50]")
        sys.exit(1)

    src = sys.argv[1]
    quality = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    info = read_bmp(src)
    dct_path = src.replace(".bmp", f"_q{quality}.dct")

    stats = compress(info["pixels"], dct_path, quality=quality)
    print(f"Compressed: ratio={stats['compression_ratio']:.3f}  "
          f"{stats['original_size']}B -> {stats['compressed_size']}B  time={stats['encode_time']:.3f}s")

    pixels_rec, dstats = decompress(dct_path)
    mse = np.mean((info["pixels"].astype(float) - pixels_rec.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    print(f"PSNR={psnr:.2f} dB  decode_time={dstats['decode_time']:.3f}s")
