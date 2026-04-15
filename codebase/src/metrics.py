"""
Metrics module: PSNR, SSIM, MSE for image quality evaluation.
"""

import numpy as np


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between two images."""
    orig = original.astype(np.float64)
    rec = reconstructed.astype(np.float64)
    return float(np.mean((orig - rec) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Returns inf if images are identical.
    """
    m = mse(original, reconstructed)
    if m == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / m))


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM).
    Uses scikit-image if available, otherwise falls back to a simple implementation.
    For multichannel images, returns mean SSIM across channels.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        if original.ndim == 3:
            return float(sk_ssim(original, reconstructed, channel_axis=2, data_range=255))
        else:
            return float(sk_ssim(original, reconstructed, data_range=255))
    except ImportError:
        return _ssim_simple(original, reconstructed)


def _ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple SSIM implementation without scikit-image."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.std()
    sigma2 = img2.std()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
    return float(num / den)


def compute_all(original: np.ndarray, reconstructed: np.ndarray,
                original_size: int, compressed_size: int,
                encode_time: float = 0.0, decode_time: float = 0.0) -> dict:
    """
    Compute all metrics and return as dict.
    original_size / compressed_size in bytes.
    """
    m = mse(original, reconstructed)
    p = psnr(original, reconstructed)
    s = ssim(original, reconstructed)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    return {
        "mse": round(m, 4),
        "psnr_db": round(p, 4) if p != float("inf") else "inf",
        "ssim": round(s, 6),
        "compression_ratio": round(ratio, 4),
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "encode_time_s": round(encode_time, 4),
        "decode_time_s": round(decode_time, 4),
    }


if __name__ == "__main__":
    # Quick self-test
    rng = np.random.default_rng(42)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-10, 10, a.shape), 0, 255).astype(np.uint8)
    print("MSE:", mse(a, b))
    print("PSNR:", psnr(a, b))
    print("SSIM:", ssim(a, b))
    print("All:", compute_all(a, b, 64*64*3, 64*64*2))
