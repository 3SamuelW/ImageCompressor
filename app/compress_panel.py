import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QRadioButton, QButtonGroup, QSlider, QFrame,
    QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_base, "codebase", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from worker import CompressWorker


def _np_to_pixmap(pixels: np.ndarray) -> QPixmap:
    """Convert numpy (H,W,3) or (H,W) uint8 to QPixmap."""
    if pixels.ndim == 2:
        h, w = pixels.shape
        img = QImage(pixels.tobytes(), w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = pixels.shape
        rgb = np.ascontiguousarray(pixels)
        img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


class DropLabel(QLabel):
    """A label that accepts BMP file drops."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._callback = None

    def set_drop_callback(self, fn):
        self._callback = fn

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(".bmp"):
                e.acceptProposedAction()
                self.setProperty("dragover", "true")
                self.style().unpolish(self)
                self.style().polish(self)

    def dragLeaveEvent(self, e):
        self.setProperty("dragover", "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, e: QDropEvent):
        self.setProperty("dragover", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        path = e.mimeData().urls()[0].toLocalFile()
        if self._callback:
            self._callback(path)


class CompressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixels = None
        self._palette = None
        self._bmp_path = None
        self._worker = None
        self._out_path = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(24)

        # ── Top row: drop zone + controls ──
        top = QHBoxLayout()
        top.setSpacing(16)

        # Left: drop zone + preview
        left = QVBoxLayout()
        left.setSpacing(8)

        self._drop = DropLabel()
        self._drop.setObjectName("drop_zone")
        self._drop.setText("Drop BMP here\nor click Browse")
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setFixedSize(240, 240)
        self._drop.set_drop_callback(self._load_bmp)
        self._drop.mousePressEvent = lambda e: self._browse()
        left.addWidget(self._drop)

        self._img_info = QLabel("")
        self._img_info.setObjectName("label_caption")
        self._img_info.setAlignment(Qt.AlignCenter)
        left.addWidget(self._img_info)

        top.addLayout(left)

        # Right: controls
        right = QVBoxLayout()
        right.setSpacing(16)
        right.setAlignment(Qt.AlignTop)

        mode_label = QLabel("Compression Mode")
        mode_label.setObjectName("label_caption")
        right.addWidget(mode_label)

        self._rb_lossless = QRadioButton("Lossless  (.hrc)")
        self._rb_lossy = QRadioButton("Lossy  (.dct)")
        self._rb_lossless.setChecked(True)
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self._rb_lossless)
        self._mode_group.addButton(self._rb_lossy)
        self._rb_lossless.toggled.connect(self._on_mode_change)
        right.addWidget(self._rb_lossless)
        right.addWidget(self._rb_lossy)

        # Quality row (hidden by default)
        self._quality_frame = QFrame()
        qf_layout = QVBoxLayout(self._quality_frame)
        qf_layout.setContentsMargins(0, 8, 0, 0)
        qf_layout.setSpacing(4)

        q_top = QHBoxLayout()
        q_lbl = QLabel("Quality")
        q_lbl.setObjectName("label_caption")
        self._q_val = QLabel("50")
        self._q_val.setObjectName("label_mono")
        q_top.addWidget(q_lbl)
        q_top.addStretch()
        q_top.addWidget(self._q_val)
        qf_layout.addLayout(q_top)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(1, 100)
        self._slider.setValue(50)
        self._slider.setFixedWidth(200)
        self._slider.valueChanged.connect(lambda v: self._q_val.setText(str(v)))
        qf_layout.addWidget(self._slider)

        q_hint = QLabel("1 = smallest file   100 = best quality")
        q_hint.setObjectName("label_helper")
        qf_layout.addWidget(q_hint)

        self._quality_frame.setVisible(False)
        right.addWidget(self._quality_frame)

        right.addStretch()

        self._btn_compress = QPushButton("Compress")
        self._btn_compress.setObjectName("btn_primary")
        self._btn_compress.setFixedHeight(48)
        self._btn_compress.setEnabled(False)
        self._btn_compress.clicked.connect(self._run_compress)
        right.addWidget(self._btn_compress)

        top.addLayout(right)
        top.addStretch()
        root.addLayout(top)

        # ── Divider ──
        div = QFrame()
        div.setObjectName("divider")
        div.setFrameShape(QFrame.HLine)
        root.addWidget(div)

        # ── Progress bar ──
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # ── Result card ──
        self._result_card = QFrame()
        self._result_card.setObjectName("card")
        self._result_card.setVisible(False)
        rc_layout = QVBoxLayout(self._result_card)
        rc_layout.setContentsMargins(16, 16, 16, 16)
        rc_layout.setSpacing(16)

        rc_title = QLabel("Compression Result")
        rc_title.setObjectName("label_heading")
        rc_layout.addWidget(rc_title)

        self._metrics_row = QHBoxLayout()
        self._metrics_row.setSpacing(32)
        rc_layout.addLayout(self._metrics_row)

        self._btn_save = QPushButton("Save Compressed File")
        self._btn_save.setObjectName("btn_secondary")
        self._btn_save.setFixedHeight(48)
        self._btn_save.setFixedWidth(220)
        self._btn_save.clicked.connect(self._save_file)
        rc_layout.addWidget(self._btn_save)

        root.addWidget(self._result_card)
        root.addStretch()

    # ── helpers ──

    def _on_mode_change(self):
        self._quality_frame.setVisible(self._rb_lossy.isChecked())

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open BMP", "", "BMP Images (*.bmp)")
        if path:
            self._load_bmp(path)

    def _load_bmp(self, path: str):
        try:
            import bmp_io
            data = bmp_io.read_bmp(path)
            self._pixels = data["pixels"]
            self._palette = data.get("palette")
            self._bmp_path = path
            # show preview
            pm = _np_to_pixmap(self._pixels).scaled(
                240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._drop.setPixmap(pm)
            h = data["height"]
            w = data["width"]
            bd = data["bit_depth"]
            self._img_info.setText(f"{w} × {h} px · {bd}-bit")
            self._btn_compress.setEnabled(True)
            self._result_card.setVisible(False)
        except Exception as ex:
            self._img_info.setText(f"Error: {ex}")

    def _run_compress(self):
        if self._pixels is None:
            return
        mode = "lossless" if self._rb_lossless.isChecked() else "lossy"
        ext = ".hrc" if mode == "lossless" else ".dct"
        base = os.path.splitext(os.path.basename(self._bmp_path))[0]
        default_name = base + ext

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Compressed File", default_name,
            f"Compressed (*{ext})"
        )
        if not out_path:
            return

        self._out_path = out_path
        self._btn_compress.setEnabled(False)
        self._progress.setVisible(True)
        self._result_card.setVisible(False)

        quality = self._slider.value()
        self._worker = CompressWorker(
            self._pixels, out_path, mode, quality, self._palette
        )
        self._worker.finished.connect(self._on_compress_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_compress_done(self, stats: dict):
        self._progress.setVisible(False)
        self._btn_compress.setEnabled(True)

        # clear old metrics
        while self._metrics_row.count():
            item = self._metrics_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        def _metric(val_str, lbl_str):
            col = QVBoxLayout()
            col.setSpacing(2)
            v = QLabel(val_str)
            v.setObjectName("metric_value")
            l = QLabel(lbl_str)
            l.setObjectName("metric_label")
            col.addWidget(v)
            col.addWidget(l)
            w = QWidget()
            w.setLayout(col)
            return w

        ratio = stats.get("compression_ratio", 0)
        orig = stats.get("original_size", 0)
        comp = stats.get("compressed_size", 0)
        enc_t = stats.get("encode_time", 0)

        self._metrics_row.addWidget(_metric(f"{ratio:.1f}×", "Compression Ratio"))
        self._metrics_row.addWidget(_metric(f"{orig/1024:.1f} KB", "Original Size"))
        self._metrics_row.addWidget(_metric(f"{comp/1024:.1f} KB", "Compressed Size"))
        self._metrics_row.addWidget(_metric(f"{enc_t:.2f} s", "Encode Time"))

        if "quality" in stats:
            self._metrics_row.addWidget(_metric(str(stats["quality"]), "Quality"))

        self._metrics_row.addStretch()
        self._result_card.setVisible(True)

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._btn_compress.setEnabled(True)
        self._img_info.setText("Compression failed — see console")
        print(msg)

    def _save_file(self):
        # File was already saved to self._out_path during compression
        # Offer copy-to dialog
        if not self._out_path or not os.path.exists(self._out_path):
            return
        ext = os.path.splitext(self._out_path)[1]
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save As", os.path.basename(self._out_path),
            f"Compressed (*{ext})"
        )
        if dest and dest != self._out_path:
            import shutil
            shutil.copy2(self._out_path, dest)
