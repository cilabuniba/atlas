from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor

class Node:
    def __init__(self, id, pos: QPointF, tipo="Generic", attributes=None, color=QColor(174, 34, 255)):
        self.id = id
        self.pos = pos
        self.tipo = tipo 
        self.attributes = attributes if attributes else {} 

        self.edges = []
        self.graphics_item = None
        self.text_item = None
        self.radius = 20
        self.color = color
        self.importance = self.attributes.get("importance", None)
        self.border_color = self.attributes.get("border_color", None)
