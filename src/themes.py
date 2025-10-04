from pathlib import Path
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
# make a light and dark theme for the app using
import qdarkstyle
from qdarkstyle.light.palette import LightPalette
from qdarkstyle.dark.palette import DarkPalette

class ThemeManager:
    def __init__(self, app: QApplication, status_label: QLabel = None):
        self.app = app
        self.label = status_label
        self.current_theme = 'light'
        self.apply_theme(self.current_theme)

    def apply_theme(self, theme: str):
        if theme == 'dark':
            self.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=DarkPalette))
            
        else:
            self.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette))
        self.current_theme = theme
        self._update_label()

    def toggle_theme(self):
        new_theme = 'dark' if self.current_theme == 'light' else 'light'
        self.apply_theme(new_theme)
        
    def _update_label(self):
        if self.label is not None:
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setMargin(10)
            self.label.setText(f"Current Theme: {self.current_theme.capitalize()} Mode")   
        

    