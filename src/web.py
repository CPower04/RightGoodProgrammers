import sys
from pathlib import Path
import pandas as pd

from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QPushButton, QStatusBar, QTableView, QToolBar, QWidget, QVBoxLayout
)
from PyQt5.QtGui import QIcon


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return 0 if self._data is None else self._data.shape[0]

    def columnCount(self, parent=None):
        return 0 if self._data is None else self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._data is None:
            return None
        if role == Qt.DisplayRole:
            return str(self._data.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if self._data is None:
            return None
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Exoplanet CSV Viewer")
        self.setGeometry(100, 100, 900, 650)

        self.table = QTableView()
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        open_file_action = QPushButton(QIcon(), "Open CSV")
        open_file_action.clicked.connect(self.open_file)
        toolbar.addWidget(open_file_action)

        self.label = QLabel("Drag and drop a CSV file or use the 'Open CSV' button.")
        self.label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setAcceptDrops(True)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(u.isLocalFile() and u.toLocalFile().lower().endswith('.csv') for u in urls):
                event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.csv'):
                self.load_csv(file_path)
            else:
                QMessageBox.warning(self, "Invalid File", "Please drop a valid CSV file.")

    
    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv *.CSV);;All Files (*)",
            options=options
        )
        if file_path:
            self.load_csv(file_path)

    
    def load_csv(self, file_path):
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        last_error = None

       
        strategies = [
            dict(comment='#', sep=',', header=0, engine='python'),
            dict(comment='#', sep=None, header=0, engine='python'),
            dict(sep=None, header=0, engine='python'),
            dict(sep=',', header=0, engine='python'),
        ]

        for enc in encodings:
            for params in strategies:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=enc,
                        dtype=str,          
                        on_bad_lines='skip',
                        **params
                    )
                    if df is None or df.shape[1] == 0:
                        raise ValueError("Parsed zero columns; trying next strategy.")
                    model = PandasModel(df)
                    self.table.setModel(model)
                    self.table.resizeColumnsToContents()
                    self.status.showMessage(
                        f"Loaded {Path(file_path).name} (encoding: {enc})", 5000
                    )
                    self.label.setText(f"Displaying contents of: {Path(file_path).name}")
                    return
                except Exception as e:
                    last_error = e
                    continue

        QMessageBox.critical(
            self,
            "Error",
            f"Failed to load CSV file after multiple strategies.\nLast error:\n{last_error}"
        )
        self.status.showMessage("Failed to load CSV file", 5000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
