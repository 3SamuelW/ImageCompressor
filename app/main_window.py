import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from compress_panel import CompressPanel
from decompress_panel import DecompressPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Compressor")
        self.setMinimumSize(800, 600)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Navigation bar ──
        navbar = QWidget()
        navbar.setObjectName("navbar")
        navbar.setFixedHeight(48)
        nav_layout = QHBoxLayout(navbar)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        logo = QLabel("▣  Image Compressor")
        logo.setObjectName("nav_logo")
        logo.setFixedHeight(48)
        nav_layout.addWidget(logo)
        nav_layout.addStretch()

        self._btn_compress_tab = QPushButton("Compress")
        self._btn_compress_tab.setObjectName("nav_btn")
        self._btn_compress_tab.setFixedHeight(48)
        self._btn_compress_tab.setProperty("active", "true")
        self._btn_compress_tab.clicked.connect(lambda: self._switch_tab(0))

        self._btn_decompress_tab = QPushButton("Decompress")
        self._btn_decompress_tab.setObjectName("nav_btn")
        self._btn_decompress_tab.setFixedHeight(48)
        self._btn_decompress_tab.setProperty("active", "false")
        self._btn_decompress_tab.clicked.connect(lambda: self._switch_tab(1))

        nav_layout.addWidget(self._btn_compress_tab)
        nav_layout.addWidget(self._btn_decompress_tab)

        root.addWidget(navbar)

        # ── Thin blue accent line under navbar ──
        accent = QFrame()
        accent.setFixedHeight(2)
        accent.setStyleSheet("background-color: #0f62fe;")
        root.addWidget(accent)

        # ── Stacked panels ──
        self._stack = QStackedWidget()
        self._compress_panel = CompressPanel()
        self._decompress_panel = DecompressPanel()
        self._stack.addWidget(self._compress_panel)
        self._stack.addWidget(self._decompress_panel)
        root.addWidget(self._stack)

    def _switch_tab(self, index: int):
        self._stack.setCurrentIndex(index)
        self._btn_compress_tab.setProperty("active", "true" if index == 0 else "false")
        self._btn_decompress_tab.setProperty("active", "true" if index == 1 else "false")
        for btn in (self._btn_compress_tab, self._btn_decompress_tab):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
