# ImageCompressor

A BMP image compression tool supporting both lossless and lossy codecs, with a desktop GUI.

## Download

**[⬇ Download ImageCompressor.exe (Windows)](https://github.com/3SamuelW/ImageCompressor/releases/tag/v0.0.0)**

No installation required — just run the exe.

## Codecs

| Mode | Format | Method |
|------|--------|--------|
| Lossless | `.hrc` | PNG prediction filter → RLE → Huffman |
| Lossy | `.dct` | YCbCr → 8×8 DCT → Quantization → Huffman |

## Usage

### GUI (recommended)

Run `ImageCompressor.exe`, then:

- **Compress tab** — drop or browse a `.bmp` file, choose lossless or lossy (with quality slider), click Compress
- **Decompress tab** — open a `.hrc` or `.dct` file, click Decompress, save result as BMP

### Run from source

Requirements: Python 3.x, `numpy`, `scipy`, `scikit-image`, `PyQt5`

```bash
cd app
python main.py
```

## Project Structure

```
ImageCompressor/
├── app/                  # GUI application (PyQt5)
│   ├── main.py
│   ├── main_window.py
│   ├── compress_panel.py
│   ├── decompress_panel.py
│   ├── worker.py
│   ├── styles.qss
│   └── assets/fonts/
└── codebase/
    ├── src/              # Core codecs
    │   ├── codec_lossless.py
    │   ├── codec_lossy.py
    │   ├── bmp_io.py
    │   ├── metrics.py
    │   └── visualize.py
    └── run_experiments.py
```
