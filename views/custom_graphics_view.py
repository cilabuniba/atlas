from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QCursor

class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.last_mouse_pos = None
        self._is_panning = False
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or \
           (self.scene().mode == "pan" and event.button() == Qt.MouseButton.LeftButton):
            self._is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or \
           (self.scene().mode == "pan" and event.button() == Qt.MouseButton.LeftButton):
            self._is_panning = False
            self.last_mouse_pos = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            event.accept()
        else:
            super().mouseReleaseEvent(event)
            
    def mouseMoveEvent(self, event):
        if self._is_panning and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            self.last_mouse_pos = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)
        
    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        oldPos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

        newPos = self.mapToScene(event.position().toPoint())
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        
    def enterEvent(self, event):
        super().enterEvent(event)
        if self.scene().mode == "pan":
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            
    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor)) 