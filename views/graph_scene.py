from PyQt6.QtWidgets import QGraphicsScene, QMenu, QColorDialog, QInputDialog, QGraphicsView, QMessageBox, QApplication
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPen, QColor
from models.node import Node
from models.edge import Edge

AVAILABLE_SHAPES = {
    "Circle": "circle",
    "Rectangle": "rect",
    "Ellipse": "ellipse",
    "Diamond": "diamond"
}

def hex_to_rgb(hex_color):
    """
    Convert HEX color string to RGB tuple.

    Example:
        hex_to_rgb("#4C78A8")
        -> (76, 120, 168)
    """
    hex_color = hex_color.lstrip("#")

    return tuple(
        int(hex_color[i:i+2], 16)
        for i in (0, 2, 4)
    )


class GraphScene(QGraphicsScene):
    graphModified = pyqtSignal()  # New signal for graph modifications
    
    def __init__(self,mode_type="Homogeneous"):
        super().__init__()
        self.mode_type = mode_type
        self.nodes = {}
        self.node_types = {}
        self.node_counter = 0
        self.selected_node = None
        self.mode = None 
        self.metrics_callback = None
        self.last_pan_pos = None
        self.moving_node = None  # Track the node being moved
        self.menu_open = False
        self.type_counters = {}
        self.legend_widget = None
    

    def set_legend_widget(self, widget):
        self.legend_widget = widget

    def update_legend(self):
        if self.legend_widget is None:
            return
        self.legend_widget.clear()
        for tipo, count in self.type_counters.items():
            self.legend_widget.addItem(f"{tipo}: {count}")

        
    def set_metrics_callback(self, callback):
        self.metrics_callback = callback
        
    def update_metrics(self):
        if self.metrics_callback:
            self.metrics_callback()
        self.graphModified.emit()  # Emit signal when graph is modified

    def add_node(self, node_id, pos, tipo="Generic", attributes=None):
        from models.node import Node
        from PyQt6.QtGui import QBrush

        node = Node(node_id, pos, tipo, attributes)


        shape = attributes.get("shape", "Circle")
        color = attributes.get("color", "#B279A2")
        color=QColor(*hex_to_rgb(color))

        if self.mode_type == "Heterogeneous":
            if tipo not in self.node_types:
                self.node_types[tipo] = {"shape": shape, "color": color}
                # self.node_types[tipo]["shape"] = tipo_shape
                # self.node_types[tipo]["color"] = color
        # if self.mode_type == "Heterogeneous" and tipo_shape in self.node_types:

        #     shape = self.node_types[tipo_shape]["shape"]
        #     color = self.node_types[tipo_shape]["color"]
        node.color = color


        from PyQt6.QtGui import QPixmap

        image_path = self.node_types[tipo].get("image") if (
            self.mode_type == "Heterogeneous" and tipo in self.node_types
        ) else None

        if image_path:
            pixmap = QPixmap(image_path).scaled(
                node.radius * 2, node.radius * 2,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            shape_item = self.addPixmap(pixmap)
            shape_item.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)
            shape_item.setPos(pos)

            shape_item.setFlags(
                shape_item.flags() |
                shape_item.GraphicsItemFlag.ItemIsSelectable |
                shape_item.GraphicsItemFlag.ItemIsMovable |
                shape_item.GraphicsItemFlag.ItemIsFocusable
            )
            shape_item.setData(0, node)

        else:
            if shape == "Rectangle":
                shape_item = self.addRect(
                    pos.x() - node.radius,
                    pos.y() - node.radius,
                    node.radius * 2,
                    node.radius * 2,
                    QPen(Qt.GlobalColor.black)
                )
                shape_item.setBrush(QBrush(color))

            elif shape == "Ellipse":
                shape_item = self.addEllipse(
                    pos.x() - node.radius * 1.5,
                    pos.y() - node.radius,
                    node.radius * 3,
                    node.radius * 2,
                    QPen(Qt.GlobalColor.black)
                )
                shape_item.setBrush(QBrush(color))

            elif shape == "Diamond":
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF
                diamond = QPolygonF([
                    QPointF(pos.x(), pos.y() - node.radius),
                    QPointF(pos.x() + node.radius, pos.y()),
                    QPointF(pos.x(), pos.y() + node.radius),
                    QPointF(pos.x() - node.radius, pos.y())
                ])

                shape_item = self.addPolygon(diamond, QPen(Qt.GlobalColor.black))
                shape_item.setBrush(QBrush(color))

            else:  # Default Circle
                shape_item = self.addEllipse(
                    pos.x() - node.radius,
                    pos.y() - node.radius,
                    node.radius * 2,
                    node.radius * 2,
                    QPen(Qt.GlobalColor.black)
                )
                shape_item.setBrush(QBrush(color))


                # --------------------------
                # Aggiungi testo e tooltip
                # --------------------------
        shape_item.setZValue(1)
        shape_item.setFlag(shape_item.GraphicsItemFlag.ItemIsSelectable)
        shape_item.setFlag(shape_item.GraphicsItemFlag.ItemIsFocusable)
        shape_item.setData(0, node)

        text = self.addText(str(node_id))
        text.setZValue(2)
        text.setDefaultTextColor(Qt.GlobalColor.black)
        text.setPos(
        pos.x() - text.boundingRect().width() / 2,
            pos.y() - text.boundingRect().height() / 2
        )

        descrizione = attributes.get("description", "") if attributes else ""
        tooltip = f"{node_id} - {descrizione}" if descrizione else node_id
        shape_item.setToolTip(tooltip)
        text.setToolTip(tooltip)

        node.graphics_item = shape_item
        node.text_item = text
        self.nodes[node_id] = node



        if self.mode_type == "Heterogeneous":
            tipo = node.tipo
            self.type_counters[tipo] = self.type_counters.get(tipo, 0) + 1
            self.update_legend()


        self.update_metrics()
        return node




    
      
    def delete_node(self, node):
        # Remove all edges connected to this node
        edges_to_remove = node.edges.copy()
        for edge in edges_to_remove:
            self.delete_edge(edge)
            
        # Remove visual items
        self.removeItem(node.graphics_item)
        self.removeItem(node.text_item)
        
        # Remove from nodes dictionary
        del self.nodes[node.id]
            
        self.menu_open = False
        self.update_metrics()

    def add_edge(self, source_node, target_node, tipo=None, attributes=None):
        if source_node == target_node:
            return

        
        if tipo is None or attributes is None:
            tipo, ok = QInputDialog.getText(None, "Edge Type", "Edge type (e.g., connects, participates):")
            if not ok or not tipo.strip():
                return

            description, ok = QInputDialog.getText(None, "Description", "Edge description:")
            if not ok:
                return

            tipo = tipo.strip()
            attributes = {"description": description.strip()}

        
        line = self.addLine(
            source_node.pos.x(), source_node.pos.y(),
            target_node.pos.x(), target_node.pos.y()
        )

        
        edge = Edge(source_node, target_node, tipo, attributes)
        edge.graphics_item = line

        # Tooltip
        tooltip = f"{tipo} - {attributes.get('description', '')}"
        line.setToolTip(tooltip)

        
        mid_x = (source_node.pos.x() + target_node.pos.x()) / 2
        mid_y = (source_node.pos.y() + target_node.pos.y()) / 2
        label = self.addText(tipo)
        label.setPos(mid_x, mid_y)
        label.setZValue(1)
        label.setToolTip(tooltip)

        edge.label_item = label

        source_node.edges.append(edge)
        target_node.edges.append(edge)

        self.update_metrics()
        return edge




    def delete_edge(self, edge):
        # Remove edge from both nodes
        if edge in edge.source.edges:
            edge.source.edges.remove(edge)
        if edge in edge.target.edges:
            edge.target.edges.remove(edge)
        
        # Remove visual item
        self.removeItem(edge.graphics_item)
        
        self.update_metrics()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.handle_right_click(event)
            return
            
        if event.button() == Qt.MouseButton.LeftButton:
            if self.menu_open:
                self.menu_open = False
                return

            pos = event.scenePos()
            
            if self.mode == "pan":
                self.last_pan_pos = pos
                for view in self.views():
                    view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            elif self.mode == "add_node":
                self.add_node(pos)
            elif self.mode == "add_edge":
                self.handle_edge_creation(pos)
            elif self.mode == "move_node":
                # Check if we clicked on a node
                items = self.items(pos)
                for item in items:
                    node = self.find_node_by_item(item)
                    if node:
                        self.moving_node = node
                        break

    def find_node_by_item(self, item):
        return item.data(0) if item and item.data(0) else None                
                
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "pan":
                self.last_pan_pos = None
                for view in self.views():
                    view.setDragMode(QGraphicsView.DragMode.NoDrag)
            elif self.mode == "move_node":
                self.moving_node = None
        
    def mouseMoveEvent(self, event):
        if self.mode == "pan" and self.last_pan_pos is not None:
            pos = event.scenePos()
            delta = pos - self.last_pan_pos
            for view in self.views():
                view.horizontalScrollBar().setValue(
                    view.horizontalScrollBar().value() - delta.x())
                view.verticalScrollBar().setValue(
                    view.verticalScrollBar().value() - delta.y())
            self.last_pan_pos = pos

        elif self.mode == "move_node" and self.moving_node:
            pos = event.scenePos()
            node = self.moving_node
            node.pos = pos
            r = node.radius
            gi = node.graphics_item

            shape_key = node.attributes.get("shape", "Circle")
            if self.mode_type == "Heterogeneous" and shape_key in self.node_types:
                shape = self.node_types[shape_key]["shape"]
                image_path = self.node_types[shape_key].get("image")
            else:
                shape = shape_key
                image_path = None

            if image_path or node.attributes.get("image"):
                from PyQt6.QtGui import QPixmap
                img_path = image_path or node.attributes.get("image")
                pixmap = QPixmap(img_path).scaled(
                    r * 2, r * 2,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                gi.setPixmap(pixmap)
                gi.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)
                gi.setPos(pos)

                node.pos = pos
            elif shape == "Ellipse":
                gi.setRect(pos.x() - r * 1.5, pos.y() - r, r * 3, r * 2)
            elif hasattr(gi, "setRect"):
                gi.setRect(pos.x() - r, pos.y() - r, r * 2, r * 2)
            elif gi.type() == 5:
                from PyQt6.QtGui import QPolygonF
                from PyQt6.QtCore import QPointF
                diamond = QPolygonF([
                    QPointF(pos.x(), pos.y() - r),
                    QPointF(pos.x() + r, pos.y()),
                    QPointF(pos.x(), pos.y() + r),
                    QPointF(pos.x() - r, pos.y())
                ])
                gi.setPolygon(diamond)

            # Sposta il testo
            text_width = node.text_item.boundingRect().width()
            text_height = node.text_item.boundingRect().height()
            node.text_item.setPos(
                pos.x() - text_width / 2,
                pos.y() - text_height / 2
            )

            # Sposta gli edge
            for edge in node.edges:
                source = edge.source.pos
                target = edge.target.pos
                edge.graphics_item.setLine(source.x(), source.y(), target.x(), target.y())

                if edge.label_item:
                    mid_x = (source.x() + target.x()) / 2
                    mid_y = (source.y() + target.y()) / 2
                    edge.label_item.setPos(mid_x, mid_y)

            self.update_metrics()




    def handle_right_click(self, event):
        pos = event.scenePos()
        items = self.items(pos)
        
        menu = QMenu()
        has_items = False
        
        for item in items:
            if isinstance(item, type(self.addEllipse(0,0,0,0))):
                has_items = True
                node = self.find_node_by_item(item)
                if node:
                    color_action = menu.addAction("Change Node Color")
                    color_action.triggered.connect(lambda: self.change_node_color(node))
                    
                    label_color_action = menu.addAction("Change Label Color")
                    label_color_action.triggered.connect(lambda: self.change_label_color(node))

                    node_size_action = menu.addAction("Change Node Size")
                    node_size_action.triggered.connect(lambda: self.change_node_size(node))

                    if self.mode_type == "Heterogeneous":
                        image_action = menu.addAction("Set Node Image")
                        image_action.triggered.connect(lambda: self.set_node_image(node))

                    
                    delete_action = menu.addAction("Delete Node")
                    delete_action.triggered.connect(lambda: self.delete_node(node))
                    
                    break
            
            elif isinstance(item, type(self.addLine(0,0,0,0))):
                has_items = True
                edge = self.find_edge_by_item(item)
                if edge:
                    delete_action = menu.addAction("Delete Edge")
                    delete_action.triggered.connect(lambda: self.delete_edge(edge))
                    
                    color_action = menu.addAction("Change Edge Color")
                    color_action.triggered.connect(lambda: self.change_edge_color(edge))

                    edge_thickness_action = menu.addAction("Change Edge Thickness")
                    edge_thickness_action.triggered.connect(lambda: self.change_edge_thickness(edge))
                    
                    break 
        
        if not has_items:
           return 
        
        if not menu.isEmpty():
            self.menu_open = True
            menu.exec(event.screenPos())


    def change_node_color(self, node):
        color = QColorDialog.getColor()
        if color.isValid():
            node.color = color
            node.graphics_item.setBrush(color)
            self.menu_open = False
            self.update_metrics()
    
    def change_edge_color(self, edge):
        color = QColorDialog.getColor()
        if color.isValid():
            edge.graphics_item.setPen(QPen(color))
            self.menu_open = False
            self.update_metrics()
    
    def change_label_color(self, node):
        color = QColorDialog.getColor()
        if color.isValid():
            node.text_item.setDefaultTextColor(color)
            self.menu_open = False
            self.update_metrics()

    def change_edge_thickness(self, edge):
        value, ok = QInputDialog.getInt(None, "Change Edge Thickness", "Enter new thickness:", min=1, max=10)
        if ok:
            pen = edge.graphics_item.pen()
            pen.setWidth(value)
            edge.graphics_item.setPen(pen)
            self.menu_open = False
            self.update_metrics()

    def change_node_size(self, node, update_menu=True):
        if update_menu:
            radius, ok = QInputDialog.getInt(
                None,
                "Change Node Size",
                "Enter new node size (radius in pixels):",
                min=20,
                max=100,
                value=node.radius
            )
            if not ok:
                return
            node.radius = radius

        radius = node.radius
        gi = node.graphics_item
        pos = node.pos

        shape_key = node.attributes.get("shape", "Circle")
        if self.mode_type == "Heterogeneous" and shape_key in self.node_types:
            shape = self.node_types[shape_key]["shape"]
            image_path = self.node_types[shape_key].get("image")
        else:
            shape = shape_key
            image_path = None

        image_path = image_path or node.attributes.get("image")
        if image_path:
            from PyQt6.QtGui import QPixmap
            pixmap = QPixmap(image_path).scaled(

                radius * 2, radius * 2,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            gi.setPixmap(pixmap)
            gi.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)
            gi.setPos(pos)
        elif shape == "Ellipse":
            gi.setRect(pos.x() - radius * 1.5, pos.y() - radius, radius * 3, radius * 2)
        elif hasattr(gi, "setRect"):
            gi.setRect(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        elif gi.type() == 5:
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            diamond = QPolygonF([
                QPointF(pos.x(), pos.y() - radius),
                QPointF(pos.x() + radius, pos.y()),
                QPointF(pos.x(), pos.y() + radius),
                QPointF(pos.x() - radius, pos.y())
            ])
            gi.setPolygon(diamond)

        # Aggiorna testo
        font = node.text_item.font()
        font.setPointSize(int(radius / 2))
        node.text_item.setFont(font)
        text_width = node.text_item.boundingRect().width()
        text_height = node.text_item.boundingRect().height()
        node.text_item.setPos(pos.x() - text_width / 2, pos.y() - text_height / 2)

        if update_menu:
            self.menu_open = False
        self.update_metrics()




    def change_nodes_size(self, value):
        for node in self.nodes.values():
            node.radius = value
            self.change_node_size(node, update_menu=False)
            




    def handle_edge_creation(self, pos):
        items = self.items(pos)
        for item in items:
            node = self.find_node_by_item(item)
            if node:
                if self.selected_node is None:
                    self.selected_node = node
                else:
                    if node != self.selected_node:
                        self.add_edge(self.selected_node, node)
                    self.selected_node = None
                break


    def find_node_by_item(self, item):
        return item.data(0) if item and item.data(0) else None

    def find_edge_by_item(self, item):
        for node in self.nodes.values():
            for edge in node.edges:
                if edge.graphics_item == item:
                    return edge
        return None


    def show_color_dialog(self, element_type):
        if element_type == "nodes" or element_type == "labels":
            has_elements = len(self.nodes) > 0
            warning_message = "There are no nodes in the graph. Please add nodes before changing their color."
        elif element_type == "edges":
            has_elements = any(len(node.edges) > 0 for node in self.nodes.values())
            warning_message = "There are no edges in the graph. Please add edges before changing their color."

        if not has_elements:
            QMessageBox.warning(
                None,
                "Error",
                warning_message 
            )
            return

        color = QColorDialog.getColor()
        if color.isValid():
            if element_type == "nodes":
                self.change_nodes_color(color)
            elif element_type == "edges":
                self.change_edges_color(color)
            elif element_type == "labels":
                self.change_labels_color(color)

    def show_thickness_dialog(self):
        edge_count = sum(len(node.edges) for node in self.nodes.values()) // 2  
        if edge_count == 0:
            QMessageBox.warning(
                None,
                "Error",
                f"There are no edges in the graph. Please add edges before changing their thickness."
            )
            return

        value, ok = QInputDialog.getInt(None, "Change Edge Thickness", "Enter new thickness:", min=1, max=10)
        if ok:
            self.change_edges_thickness(value) 

    def show_node_size_dialog(self):
        if not self.nodes:
            QMessageBox.warning(
                None,
                "Error",
                "There are no nodes in the graph. Please add nodes before changing their size."
            )
            return

        value, ok = QInputDialog.getInt(
            None,
            "Change Node Size",
            "Enter new node size (radius in pixels):",
            min=20,
            max=100,
            value=20
        )

        if ok:
            self.change_nodes_size(value)


    def change_nodes_color(self, color):
        for node in self.nodes.values():
            node.color = color
            node.graphics_item.setBrush(color)
        self.menu_open = False
        self.update_metrics()

    def change_edges_color(self, color):
        for node in self.nodes.values():
            for edge in node.edges:
                edge.graphics_item.setPen(QPen(color))
        self.menu_open = False
        self.update_metrics()

    def change_labels_color(self, color):
        for node in self.nodes.values():
            node.text_item.setDefaultTextColor(color)
        self.menu_open = False
        self.update_metrics()

    def change_edges_thickness(self, value):
        for node in self.nodes.values():
            for edge in node.edges:
                pen = edge.graphics_item.pen()
                pen.setWidth(value)
                edge.graphics_item.setPen(pen)
        self.menu_open = False
        self.update_metrics()


    def clear_all(self):
        self.clear()
        self.nodes.clear()
        self.node_counter = 0
        self.selected_node = None
        self.update_metrics() 





    def set_node_image(self, node):
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtGui import QPixmap

        file_name, _ = QFileDialog.getOpenFileName(
            None, "Choose Image", "", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_name:

            if node.graphics_item:
                self.removeItem(node.graphics_item)

            pixmap = QPixmap(file_name).scaled(
                node.radius * 2, node.radius * 2,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            pixmap_item = self.addPixmap(pixmap)
            pixmap_item.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)
            pixmap_item.setPos(node.pos)
            pixmap_item.setFlags(
                pixmap_item.flags() |
                pixmap_item.GraphicsItemFlag.ItemIsSelectable |
                pixmap_item.GraphicsItemFlag.ItemIsMovable |
                pixmap_item.GraphicsItemFlag.ItemIsFocusable
            )
            pixmap_item.setData(0, node)
            node.graphics_item = pixmap_item
            node.attributes["image"] = file_name


            node.text_item.setZValue(2)
            node.graphics_item.setZValue(1)

            self.update_metrics()
