from PyQt6.QtWidgets import (QPlainTextEdit, QWidget, QVBoxLayout, QPushButton, 
                            QMessageBox, QHBoxLayout, QLabel, QComboBox)
from PyQt6.QtCore import pyqtSignal

class CodeEditor(QWidget):
    codeChanged = pyqtSignal(str)
    importRequested = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Preview", "Import"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        self.layout.addLayout(mode_layout)
        
        # Code editor
        self.editor = QPlainTextEdit()
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                background-color: #2b2b2b;
                color: #a9b7c6;
            }
        """)
        self.editor.textChanged.connect(self._on_text_changed)
        self.layout.addWidget(self.editor)
        
        # Import button
        self.import_button = QPushButton("Import Graph from Code")
        self.import_button.clicked.connect(self._on_import_clicked)
        self.layout.addWidget(self.import_button)
        
        # Internal flags
        self._updating = False
        
        # Set initial mode
        self._on_mode_changed("Preview")
        
    def setPlainText(self, text):
        if self.mode_combo.currentText() == "Preview" and not self._updating:
            self._updating = True
            current = self.editor.toPlainText()
            if current != text:
                self.editor.setPlainText(text)
            self._updating = False
        
    def toPlainText(self):
        return self.editor.toPlainText()
        
    def setReadOnly(self, readonly):
        self.editor.setReadOnly(readonly)
        
    def _on_mode_changed(self, mode):
        if mode == "Preview":
            self.editor.setReadOnly(True)
            self.import_button.setVisible(False)
            if not self._updating:
                self.codeChanged.emit(self.editor.toPlainText())
        else:
            self.editor.setReadOnly(False)
            self.import_button.setVisible(True)
            self.editor.clear()
            self.editor.setPlaceholderText("Paste your graph code here...")
        
    def _on_text_changed(self):
        if not self._updating and self.mode_combo.currentText() == "Preview":
            self.codeChanged.emit(self.editor.toPlainText())
        
    def _on_import_clicked(self):
        code = self.editor.toPlainText()
        try:
            # Basic validation
            if not any(lib in code for lib in ['networkx', 'igraph', 'pyvis', 'graph_tool', 'pygraphviz', 'dgl', 'snap']):
                raise ValueError("No supported graph library found in code")
            
            self.importRequested.emit(code)
            self.mode_combo.setCurrentText("Preview")
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e)) 