"""
Generate 5 test BMP images for the Image Compressor experiments.
Images: natural-like, texture, geometric, gradient, low-color.
All images are 256x256 24-bit BMP.

NOTE: Images are synthetically generated (no external downloads needed).
      For a real experiment, replace with actual photographs.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from bmp_io import write_bmp

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 256
rng = np.random.default_rng(42)


def make_natural(size=SIZE) -> np.ndarray:
    """Simulate a natural scene: smooth gradients + noise (like a blurred photo)."""
    from scipy.ndimage import gaussian_filter
    base = rng.integers(30, 220, (size, size, 3), dtype=np.uint8).astype(np.float32)
    # Add large-scale structure
    for c in range(3):
        base[:, :, c] += np.linspace(0, 60, size).reshape(-1, 1) * (c + 1) / 3
    smooth = np.stack([gaussian_filter(base[:, :, c], sigma=15) for c in range(3)], axis=2)
    return np.clip(smooth, 0, 255).astype(np.uint8)


def make_texture(size=SIZE) -> np.ndarray:
    """Repeating texture pattern (checkerboard + sine waves)."""
    x = np.linspace(0, 8 * np.pi, size)
    y = np.linspace(0, 8 * np.pi, size)
    xx, yy = np.meshgrid(x, y)
    R = ((np.sin(xx) * np.cos(yy) + 1) * 127).astype(np.uint8)
    G = ((np.sin(xx * 1.3) * np.sin(yy * 0.7) + 1) * 127).astype(np.uint8)
    B = ((np.cos(xx * 0.5 + yy * 1.5) + 1) * 127).astype(np.uint8)
    return np.stack([R, G, B], axis=2)


def make_geometric(size=SIZE) -> np.ndarray:
    """Geometric shapes: circles, rectangles on solid background."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 240  # light gray bg
    # Red rectangle
    img[30:80, 30:120] = [220, 50, 50]
    # Blue circle
    cx, cy, r = 180, 80, 50
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
    img[mask] = [50, 80, 200]
    # Green triangle (filled via rasterization)
    for row in range(120, 220):
        left = 60 + (row - 120) // 2
        right = 200 - (row - 120) // 2
        if left < right:
            img[row, left:right] = [50, 180, 80]
    # Yellow diamond
    for row in range(150, 230):
        d = abs(row - 190)
        img[row, 170 - d: 170 + d] = [230, 200, 30]
    return img


def make_gradient(size=SIZE) -> np.ndarray:
    """Smooth color gradient (horizontal R, vertical G, diagonal B)."""
    r = np.linspace(0, 255, size).reshape(1, -1).repeat(size, axis=0).astype(np.uint8)
    g = np.linspace(0, 255, size).reshape(-1, 1).repeat(size, axis=1).astype(np.uint8)
    b = ((r.astype(np.float32) + g.astype(np.float32)) / 2).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def make_low_color(size=SIZE) -> np.ndarray:
    """Low-color image: only 8 distinct colors (like a simple cartoon)."""
    palette = np.array([
        [255, 255, 255],  # white
        [0,   0,   0  ],  # black
        [255, 0,   0  ],  # red
        [0,   255, 0  ],  # green
        [0,   0,   255],  # blue
        [255, 255, 0  ],  # yellow
        [255, 0,   255],  # magenta
        [0,   255, 255],  # cyan
    ], dtype=np.uint8)
    # Divide image into 8 horizontal bands
    img = np.zeros((size, size, 3), dtype=np.uint8)
    band = size // 8
    for i in range(8):
        img[i*band:(i+1)*band, :] = palette[i]
    return img


images = {
    "natural.bmp":   make_natural(),
    "texture.bmp":   make_texture(),
    "geometric.bmp": make_geometric(),
    "gradient.bmp":  make_gradient(),
    "low_color.bmp": make_low_color(),
}

for name, pixels in images.items():
    path = OUT_DIR / name
    write_bmp(str(path), pixels)
    print(f"Written: {path}  shape={pixels.shape}")

print(f"\nAll {len(images)} test BMP images written to {OUT_DIR}")
