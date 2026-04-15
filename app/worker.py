import sys
import os

# Ensure codebase/src is importable when running as EXE or script
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_base, "codebase", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from PyQt5.QtCore import QThread, pyqtSignal
import traceback


class CompressWorker(QThread):
    """Run compression in a background thread."""
    finished = pyqtSignal(dict)   # emits result dict on success
    error = pyqtSignal(str)       # emits error message on failure

    def __init__(self, pixels, out_path, mode, quality=50, palette=None):
        super().__init__()
        self.pixels = pixels
        self.out_path = out_path
        self.mode = mode          # "lossless" or "lossy"
        self.quality = quality
        self.palette = palette

    def run(self):
        try:
            if self.mode == "lossless":
                import codec_lossless
                stats = codec_lossless.compress(self.pixels, self.out_path, self.palette)
            else:
                import codec_lossy
                stats = codec_lossy.compress(self.pixels, self.out_path, self.quality)
            self.finished.emit(stats)
        except Exception:
            self.error.emit(traceback.format_exc())


class DecompressWorker(QThread):
    """Run decompression in a background thread."""
    finished = pyqtSignal(object, object, dict)  # pixels, palette, stats
    error = pyqtSignal(str)

    def __init__(self, in_path, mode):
        super().__init__()
        self.in_path = in_path
        self.mode = mode   # "lossless" or "lossy"

    def run(self):
        try:
            if self.mode == "lossless":
                import codec_lossless
                pixels, palette, stats = codec_lossless.decompress(self.in_path)
                self.finished.emit(pixels, palette, stats)
            else:
                import codec_lossy
                pixels, stats = codec_lossy.decompress(self.in_path)
                self.finished.emit(pixels, None, stats)
        except Exception:
            self.error.emit(traceback.format_exc())
