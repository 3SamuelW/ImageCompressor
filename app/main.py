import sys
import os

# When running as a PyInstaller bundle, _MEIPASS holds the temp extraction dir.
# We need to add codebase/src to sys.path so codec imports work.
if getattr(sys, "frozen", False):
    _bundle_dir = sys._MEIPASS
else:
    _bundle_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_src = os.path.join(_bundle_dir, "codebase", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase, QFont
from PyQt5.QtCore import Qt

from main_window import MainWindow


def _load_fonts(app_dir: str):
    fonts_dir = os.path.join(app_dir, "assets", "fonts")
    for fname in [
        "IBMPlexSans-Light.ttf",
        "IBMPlexSans-Regular.ttf",
        "IBMPlexSans-SemiBold.ttf",
        "IBMPlexMono-Regular.ttf",
    ]:
        path = os.path.join(fonts_dir, fname)
        if os.path.exists(path):
            QFontDatabase.addApplicationFont(path)


def _load_stylesheet(app_dir: str) -> str:
    qss_path = os.path.join(app_dir, "styles.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def main():
    # Enable HiDPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Resolve app directory (works both frozen and script)
    if getattr(sys, "frozen", False):
        app_dir = os.path.join(sys._MEIPASS, "app")
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))

    _load_fonts(app_dir)

    # Set default font
    default_font = QFont("IBM Plex Sans", 10)
    app.setFont(default_font)

    # Apply stylesheet
    stylesheet = _load_stylesheet(app_dir)
    if stylesheet:
        app.setStyleSheet(stylesheet)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
