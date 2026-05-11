from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen

class Edge:
    def __init__(self, source, target, tipo="Generic", attributes=None):
        self.source = source
        self.target = target
        self.tipo = tipo
        self.attributes = attributes if attributes else {}
        self.graphics_item = None 
        self.label_item = None
        self.color = QColor(Qt.GlobalColor.black)