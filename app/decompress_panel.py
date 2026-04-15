import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QProgressBar, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_base, "codebase", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from worker import DecompressWorker


def _np_to_pixmap(pixels: np.ndarray) -> QPixmap:
    if pixels.ndim == 2:
        h, w = pixels.shape
        img = QImage(pixels.tobytes(), w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = pixels.shape
        rgb = np.ascontiguousarray(pixels)
        img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def _detect_mode(path: str) -> str:
    """Read magic bytes to determine codec type."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"HRC2":
        return "lossless"
    elif magic == b"DCT2":
        return "lossy"
    return "unknown"


class DecompressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._in_path = None
        self._mode = None
        self._pixels = None
        self._palette = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scroll area so nothing gets clipped
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)

        root = QVBoxLayout(inner)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(24)

        # ── File selection row ──
        file_row = QHBoxLayout()
        file_row.setSpacing(16)

        self._file_label = QLabel("No file selected")
        self._file_label.setObjectName("label_mono")
        self._file_label.setSizePolicy(
            self._file_label.sizePolicy().horizontalPolicy(),
            self._file_label.sizePolicy().verticalPolicy()
        )
        file_row.addWidget(self._file_label, 1)

        btn_browse = QPushButton("Browse File")
        btn_browse.setObjectName("btn_ghost")
        btn_browse.setFixedHeight(40)
        btn_browse.clicked.connect(self._browse)
        file_row.addWidget(btn_browse)

        root.addLayout(file_row)

        # ── File info card ──
        self._info_card = QFrame()
        self._info_card.setObjectName("card")
        self._info_card.setVisible(False)
        info_layout = QHBoxLayout(self._info_card)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_layout.setSpacing(32)

        self._lbl_format = self._make_metric("—", "Format")
        self._lbl_size = self._make_metric("—", "File Size")
        self._lbl_mode_tag = QLabel("")
        self._lbl_mode_tag.setObjectName("tag_info")
        self._lbl_mode_tag.setAlignment(Qt.AlignCenter)

        info_layout.addWidget(self._lbl_format)
        info_layout.addWidget(self._lbl_size)
        info_layout.addWidget(self._lbl_mode_tag)
        info_layout.addStretch()

        root.addWidget(self._info_card)

        # ── Decompress button ──
        self._btn_decompress = QPushButton("Decompress")
        self._btn_decompress.setObjectName("btn_primary")
        self._btn_decompress.setFixedHeight(48)
        self._btn_decompress.setFixedWidth(200)
        self._btn_decompress.setEnabled(False)
        self._btn_decompress.clicked.connect(self._run_decompress)
        root.addWidget(self._btn_decompress)

        # ── Progress bar ──
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # ── Divider ──
        div = QFrame()
        div.setObjectName("divider")
        div.setFrameShape(QFrame.HLine)
        root.addWidget(div)

        # ── Result area ──
        self._result_card = QFrame()
        self._result_card.setObjectName("card")
        self._result_card.setVisible(False)
        rc_layout = QVBoxLayout(self._result_card)
        rc_layout.setContentsMargins(16, 16, 16, 16)
        rc_layout.setSpacing(16)

        rc_title = QLabel("Decompression Result")
        rc_title.setObjectName("label_heading")
        rc_layout.addWidget(rc_title)

        result_row = QHBoxLayout()
        result_row.setSpacing(32)

        # Preview — scales with window, min 200px, max 400px
        self._preview = QLabel()
        self._preview.setMinimumSize(200, 200)
        self._preview.setMaximumSize(400, 400)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.setObjectName("drop_zone")
        self._preview.setAlignment(Qt.AlignCenter)
        result_row.addWidget(self._preview, 1)

        # Stats
        stats_col = QVBoxLayout()
        stats_col.setSpacing(16)
        stats_col.setAlignment(Qt.AlignTop)

        self._metrics_col = QVBoxLayout()
        self._metrics_col.setSpacing(8)
        stats_col.addLayout(self._metrics_col)
        stats_col.addStretch()

        self._btn_save = QPushButton("Save as BMP")
        self._btn_save.setObjectName("btn_secondary")
        self._btn_save.setFixedHeight(48)
        self._btn_save.setFixedWidth(180)
        self._btn_save.clicked.connect(self._save_bmp)
        stats_col.addWidget(self._btn_save)

        result_row.addLayout(stats_col)
        result_row.addStretch()
        rc_layout.addLayout(result_row)

        root.addWidget(self._result_card)
        root.addStretch()

    def _make_metric(self, val: str, lbl: str) -> QWidget:
        col = QVBoxLayout()
        col.setSpacing(2)
        v = QLabel(val)
        v.setObjectName("metric_value")
        l = QLabel(lbl)
        l.setObjectName("metric_label")
        col.addWidget(v)
        col.addWidget(l)
        w = QWidget()
        w.setLayout(col)
        return w

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Compressed File", "",
            "Compressed Files (*.hrc *.dct);;All Files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        self._in_path = path
        self._mode = _detect_mode(path)

        fname = os.path.basename(path)
        self._file_label.setText(fname)

        size_kb = os.path.getsize(path) / 1024
        ext = os.path.splitext(path)[1].upper().lstrip(".")

        # update info card
        self._lbl_format.findChild(QLabel, "").deleteLater() if False else None
        # rebuild info card labels
        for w in self._info_card.findChildren(QLabel):
            if w.objectName() in ("metric_value", "metric_label"):
                pass  # keep structure, just update text

        # simpler: just update via stored refs
        fmt_labels = [c for c in self._lbl_format.findChildren(QLabel)]
        if len(fmt_labels) >= 1:
            fmt_labels[0].setText(ext)
        size_labels = [c for c in self._lbl_size.findChildren(QLabel)]
        if len(size_labels) >= 1:
            size_labels[0].setText(f"{size_kb:.1f} KB")

        if self._mode == "lossless":
            self._lbl_mode_tag.setText("  Lossless · HRC2  ")
            self._lbl_mode_tag.setObjectName("tag_success")
        elif self._mode == "lossy":
            self._lbl_mode_tag.setText("  Lossy · DCT2  ")
            self._lbl_mode_tag.setObjectName("tag_info")
        else:
            self._lbl_mode_tag.setText("  Unknown Format  ")
            self._lbl_mode_tag.setObjectName("tag_error")

        self._lbl_mode_tag.style().unpolish(self._lbl_mode_tag)
        self._lbl_mode_tag.style().polish(self._lbl_mode_tag)

        self._info_card.setVisible(True)
        self._btn_decompress.setEnabled(self._mode in ("lossless", "lossy"))
        self._result_card.setVisible(False)

    def _run_decompress(self):
        if not self._in_path or self._mode not in ("lossless", "lossy"):
            return
        self._btn_decompress.setEnabled(False)
        self._progress.setVisible(True)
        self._result_card.setVisible(False)

        self._worker = DecompressWorker(self._in_path, self._mode)
        self._worker.finished.connect(self._on_decompress_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_decompress_done(self, pixels, palette, stats: dict):
        self._progress.setVisible(False)
        self._btn_decompress.setEnabled(True)
        self._pixels = pixels
        self._palette = palette

        # preview — scale to actual widget size
        raw_pm = _np_to_pixmap(pixels)
        self._preview._raw_pm = raw_pm
        size = self._preview.size()
        pm = raw_pm.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._preview.setPixmap(pm)

        # clear old metrics
        while self._metrics_col.count():
            item = self._metrics_col.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        def _row(lbl, val):
            row = QHBoxLayout()
            l = QLabel(lbl)
            l.setObjectName("metric_label")
            v = QLabel(val)
            v.setObjectName("label_mono")
            row.addWidget(l)
            row.addWidget(v)
            row.addStretch()
            w = QWidget()
            w.setLayout(row)
            return w

        dec_t = stats.get("decode_time", 0)
        self._metrics_col.addWidget(_row("Decode Time", f"{dec_t:.3f} s"))

        if pixels.ndim == 3:
            h, w, _ = pixels.shape
            self._metrics_col.addWidget(_row("Dimensions", f"{w} × {h} px"))
        else:
            h, w = pixels.shape
            self._metrics_col.addWidget(_row("Dimensions", f"{w} × {h} px"))

        if "quality" in stats:
            self._metrics_col.addWidget(_row("Quality", str(stats["quality"])))

        self._result_card.setVisible(True)

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._btn_decompress.setEnabled(True)
        self._file_label.setText("Decompression failed — see console")
        print(msg)

    def _save_bmp(self):
        if self._pixels is None:
            return
        base = os.path.splitext(os.path.basename(self._in_path))[0]
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save BMP", base + "_decoded.bmp", "BMP Image (*.bmp)"
        )
        if not dest:
            return
        try:
            import bmp_io
            bmp_io.write_bmp(dest, self._pixels, self._palette)
        except Exception as ex:
            self._file_label.setText(f"Save failed: {ex}")
