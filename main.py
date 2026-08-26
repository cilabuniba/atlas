import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QComboBox,
    QMessageBox,
    QInputDialog,
    QMenu,
    QFileDialog,
    QTextEdit,
    QSplitter,
    QTabWidget,
    QLabel,
    QHBoxLayout,
    QPlainTextEdit,
    QToolButton
)
from PyQt6.QtCore import Qt, QPointF
from random import randint
from PyQt6.QtGui import QPen, QColor, QPainter, QIcon, QFont

from models import node, edge 
from views.graph_scene import GraphScene
from views.custom_graphics_view import CustomGraphicsView
from widgets.code_editor import CodeEditor
from widgets.metrics_panel import MetricsPanel
from utils import exporters
from utils.code_importer import GraphImporter


class GraphEditor(QMainWindow):
    def __init__(self, graph_mode="Homogeneous"):
        super().__init__()

        self.graph_mode = graph_mode


        self.setWindowTitle("Graph Editor")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon("resources/icon.png"))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.main_splitter)

        # LEFT PANEL
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        self.btn_layout = QHBoxLayout()
        btn_layout = self.btn_layout
        btn_add_node = QPushButton("Add Node")
        btn_add_node.clicked.connect(self.prompt_new_node)
        btn_add_edge = QPushButton("Add Edge")
        btn_add_edge.clicked.connect(lambda: self.set_mode("add_edge"))
        btn_move_node = QPushButton("Move Node")
        btn_move_node.clicked.connect(lambda: self.set_mode("move_node"))
        btn_pan = QPushButton("Pan Mode")
        btn_pan.clicked.connect(lambda: self.set_mode("pan"))
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_all)
        btn_back = QPushButton("Back to Mode Selection")
        btn_back.clicked.connect(self._restart_requested)
        btn_layout.addWidget(btn_back)



        btn_define_node_type = QPushButton("Define Node Types")
        btn_define_node_type.clicked.connect(self.define_node_types)
        if self.graph_mode == "Heterogeneous":
            btn_layout.addWidget(btn_define_node_type)

        btn_layout.addWidget(btn_add_node)
        btn_layout.addWidget(btn_add_edge)
        btn_layout.addWidget(btn_move_node)
        btn_layout.addWidget(btn_pan)
        btn_layout.addWidget(btn_clear)
        left_layout.addLayout(btn_layout)

        self.scene = GraphScene(mode_type=graph_mode)
        self.view = CustomGraphicsView(self.scene)
        left_layout.addWidget(self.view)
        self.scene.graphModified.connect(self.update_code_preview)
        self.main_splitter.addWidget(left_widget)
        

        # RIGHT PANEL
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)


        btn_right_layout_change = QHBoxLayout()

        btn_change_colors = QToolButton()
        btn_change_colors.setText("Change Colors")
        btn_change_colors.setFixedWidth(140)
        color_menu = QMenu()
        node_color_action = color_menu.addAction("Change All Nodes Color")
        edge_color_action = color_menu.addAction("Change All Edges Color")
        label_color_action = color_menu.addAction("Change All Labels Color")
        node_color_action.triggered.connect(lambda: self.scene.show_color_dialog("nodes"))
        edge_color_action.triggered.connect(lambda: self.scene.show_color_dialog("edges"))
        label_color_action.triggered.connect(lambda: self.scene.show_color_dialog("labels"))
        btn_change_colors.setMenu(color_menu)
        btn_change_colors.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        btn_change_thickness = QPushButton("Change Edges Thickness")
        btn_change_thickness.clicked.connect(lambda: self.scene.show_thickness_dialog())

        btn_change_node_size = QPushButton("Change Nodes Size")
        btn_change_node_size.clicked.connect(lambda: self.scene.show_node_size_dialog())

        btn_right_layout_change.addWidget(btn_change_colors)
        btn_right_layout_change.addWidget(btn_change_thickness)
        btn_right_layout_change.addWidget(btn_change_node_size)
        right_layout.addLayout(btn_right_layout_change)



        export_layout = QHBoxLayout()
        self.export_combo = QComboBox()
        if self.graph_mode == "Heterogeneous":
            self.export_combo.addItems(["DGL", "PyG"])
        else:
            self.export_combo.addItems(["NetworkX", "igraph", "PyVis", "Graph-tool", "PyGraphviz", "DGL", "SNAP", "PyG"])

        self.export_combo.currentTextChanged.connect(self.update_code_preview)

        btn_export_clipboard = QPushButton("Copy to Clipboard")
        btn_export_clipboard.clicked.connect(lambda: self.export_graph(to_clipboard=True))
        btn_export_file = QPushButton("Save to File")
        btn_export_file.clicked.connect(lambda: self.export_graph(to_clipboard=False))

        export_layout.addWidget(self.export_combo)
        export_layout.addWidget(btn_export_clipboard)
        export_layout.addWidget(btn_export_file)
        right_layout.addLayout(export_layout)

        self.tabs = QTabWidget()
        self.code_editor = CodeEditor()
        self.code_editor.codeChanged.connect(self._on_code_changed)
        self.code_editor.importRequested.connect(self._on_import_requested)
        self.code_editor.mode_combo.currentTextChanged.connect(self._on_editor_mode_changed)
        self.tabs.addTab(self.code_editor, "Code Preview/Import")

        self.metrics_panel = MetricsPanel(self.scene)

        if self.graph_mode == "Heterogeneous":
            from PyQt6.QtWidgets import QListWidget
            self.legend_widget = QListWidget()
            self.scene.set_legend_widget(self.legend_widget)
            self.tabs.addTab(self.legend_widget, "Legend")

        self.tabs.addTab(self.metrics_panel, "Metrics")
        right_layout.addWidget(self.tabs)

        self.main_splitter.addWidget(right_widget)
        self.scene.set_metrics_callback(self.metrics_panel.update_metrics)

        self.update_code_preview()
        self.main_splitter.setSizes([700, 500])


   
    
    def switch_mode(self, new_mode):
        if new_mode == self.graph_mode:
            return

        confirm = QMessageBox.question(
            self,
            "Switch Mode",
            f"Switch to {new_mode} mode?\nAll current nodes will be cleared.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.No:
            self.mode_selector.blockSignals(True)
            self.mode_selector.setCurrentText(self.graph_mode)
            self.mode_selector.blockSignals(False)
            return

        self.graph_mode = new_mode
        self.scene.clear_all()
        self.scene.mode_type = new_mode

        # Aggiorna la combo esportazione
        self.export_combo.clear()
        if new_mode == "Heterogeneous":
            self.export_combo.addItems(["DGL", "PyG"])
            # Aggiungi la leggenda se non c’è
            if not hasattr(self, "legend_widget"):
                from PyQt6.QtWidgets import QListWidget
                self.legend_widget = QListWidget()
                self.scene.set_legend_widget(self.legend_widget)
                self.tabs.insertTab(1, self.legend_widget, "Legend")
        else:
            self.export_combo.addItems(["NetworkX", "igraph", "PyVis", "Graph-tool", "PyGraphviz", "DGL", "SNAP", "PyG"])
            # Rimuovi la leggenda se presente
            if hasattr(self, "legend_widget"):
                index = self.tabs.indexOf(self.legend_widget)
                if index != -1:
                    self.tabs.removeTab(index)
                self.legend_widget = None
                self.scene.set_legend_widget(None)

        self.update_code_preview()

        if new_mode == "Heterogeneous":
            if not hasattr(self, "btn_define_node_type"):
                self.btn_define_node_type = QPushButton("Define Node Types")
                self.btn_define_node_type.clicked.connect(self.define_node_types)
                self.btn_layout.insertWidget(1, self.btn_define_node_type)
        else:
            if hasattr(self, "btn_define_node_type"):
                self.btn_define_node_type.setParent(None)
                del self.btn_define_node_type


    def define_node_types(self):
        from PyQt6.QtWidgets import (
            QDialog, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox,
            QColorDialog, QPushButton, QFileDialog
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Define Node Type")
        layout = QFormLayout(dialog)

        name_input = QLineEdit()
        shape_combo = QComboBox()
        shape_combo.addItems(["Circle", "Rectangle", "Ellipse", "Diamond"])

        color_btn = QPushButton("Choose Color")
        selected_color = [QColor(174, 34, 255)]

        def choose_color():
            color = QColorDialog.getColor()
            if color.isValid():
                selected_color[0] = color
                color_btn.setStyleSheet(f"background-color: {color.name()}")

        color_btn.clicked.connect(choose_color)

        image_path = [None]
        image_btn = QPushButton("Choose Image (optional)")

        def choose_image():
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Choose Image", "", "Image Files (*.png *.jpg *.bmp)"
            )
            if file_name:
                image_path[0] = file_name
                image_btn.setText(f"Image Selected ✓")

        image_btn.clicked.connect(choose_image)

        layout.addRow("Type Name:", name_input)
        layout.addRow("Shape (fallback if no image):", shape_combo)
        layout.addRow("Color (used if no image):", color_btn)
        layout.addRow("Image:", image_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            type_name = name_input.text().strip()
            shape = shape_combo.currentText()

            if not type_name:
                return

            self.scene.node_types[type_name] = {
                "shape": shape,
                "color": selected_color[0],
                "image": image_path[0]
            }

            QMessageBox.information(self, "Success", f"Node type '{type_name}' added!")


    def prompt_new_node(self):
        from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QComboBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Node")
        layout = QFormLayout(dialog)

        id_input = QLineEdit()
        desc_input = QLineEdit()

        layout.addRow("Node ID:", id_input)
        layout.addRow("Description:", desc_input)

        shape_combo = QComboBox()
        
        if self.graph_mode == "Heterogeneous":

            if self.scene.node_types:
                shape_combo.addItems(self.scene.node_types.keys())
            else:
                shape_combo.addItems(["Default"])
            layout.addRow("Type:", shape_combo)

        else:

            shape_combo.addItems(["Circle", "Rectangle", "Ellipse", "Diamond"])
            layout.addRow("Shape:", shape_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            node_id = id_input.text().strip()
            desc = desc_input.text().strip()
            shape_or_type = shape_combo.currentText()

            if not node_id or not desc:
                return

            center = self.view.mapToScene(self.view.viewport().rect().center())
            offset_x = randint(-100, 100)
            offset_y = randint(-100, 100)
            pos = QPointF(center.x() + offset_x, center.y() + offset_y)

            if self.graph_mode == "Heterogeneous":
                attributes = {
                    "description": desc,
                    "shape": shape_or_type  
                }
                tipo = shape_or_type
            else:
                attributes = {
                    "description": desc,
                    "shape": shape_or_type  
                }
                tipo = "Generic"  

            self.scene.add_node(node_id.strip(), pos, tipo, attributes)
   





    def _on_code_changed(self, code):
        if self.code_editor.mode_combo.currentText() == "Preview":
            if not self.code_editor._updating:
                self.update_code_preview()

    def _on_editor_mode_changed(self, mode):
        if mode == "Preview":
            self.update_code_preview()
            self.export_combo.setEnabled(True)
        else:
            self.export_combo.setEnabled(False)

    def _on_import_requested(self, code):
        try:
            metrics_file = None
            reply = QMessageBox.question(self, "Metrics", "Do you want to save performance metrics to a JSON file?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                metrics_file, _ = QFileDialog.getSaveFileName(self, "Save Metrics", "import_metrics.json", "JSON files (*.json)")
            
            GraphImporter.import_from_code(code, self.scene, metrics_file)
            self.update_code_preview()
            QMessageBox.information(self, "Success", "Graph imported successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))
            self.code_editor.mode_combo.setCurrentText("Import")

    def update_code_preview(self):
        library = self.export_combo.currentText()
        if library == "NetworkX":
            code = exporters.export_networkx(self.scene)
        elif library == "igraph":
            code = exporters.export_igraph(self.scene)
        elif library == "PyVis":
            code = exporters.export_pyvis(self.scene)
        elif library == "PyGraphviz":
            code = exporters.export_pygraphviz(self.scene)
        elif library == "Graph-tool":
            code = exporters.export_graphtool(self.scene)
        elif library == "DGL":
            code = exporters.export_dgl(self.scene)
        elif library == "SNAP":
            code = exporters.export_snap(self.scene)
        elif library == "PyG":
            code = exporters.export_pyg(self.scene)


        self.code_editor.setPlainText(code)

    def set_mode(self, mode):
        self.scene.mode = mode
        self.scene.selected_node = None

        if mode == "move_node":
            self.view.setCursor(Qt.CursorShape.OpenHandCursor)
        elif mode == "pan":
            self.view.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif mode == "add_node" or mode == "add_edge":
            self.view.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == "change_colors":
            self.view.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.view.setCursor(Qt.CursorShape.ArrowCursor)

    def clear_all(self):
        reply = QMessageBox.question(
            self,
            "Clear All",
            "Are you sure you want to clear everything?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.scene.clear_all()
            self.update_code_preview()


    def export_graph(self, to_clipboard=True):
        current_tab = self.tabs.currentWidget()

        if isinstance(current_tab, CodeEditor):
            code = self.code_editor.toPlainText()
            if to_clipboard:
                clipboard = QApplication.clipboard()
                clipboard.setText(code)
                QMessageBox.information(self, "Success", "Code copied to clipboard!")
            else:
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Code",
                    f"graph_{self.export_combo.currentText().lower()}.py",
                    "Python files (*.py);;All Files (*.*)",
                )
                if file_name:
                    with open(file_name, "w") as f:
                        f.write(code)
                    QMessageBox.information(self, "Success", f"Code saved to {file_name}")

        elif isinstance(current_tab, MetricsPanel):
            metrics_text = self.get_metrics_text()
            if to_clipboard:
                clipboard = QApplication.clipboard()
                clipboard.setText(metrics_text)
                QMessageBox.information(self, "Success", "Metrics copied to clipboard!")
            else:
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Metrics",
                    "graph_metrics.txt",
                    "Text files (*.txt);;All Files (*.*)",
                )
                if file_name:
                    with open(file_name, "w") as f:
                        f.write(metrics_text)
                    QMessageBox.information(self, "Success", f"Metrics saved to {file_name}")

    def get_metrics_text(self):
        metrics = []
        metrics.append("Graph Metrics Report")
        metrics.append("===================\n")

        num_nodes = len(self.scene.nodes)
        num_edges = sum(len(node.edges) for node in self.scene.nodes.values()) // 2

        metrics.append(f"Number of nodes: {num_nodes}")
        metrics.append(f"Number of edges: {num_edges}")

        if num_nodes > 1:
            density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
            metrics.append(f"Density: {density:.3f}")

        if num_nodes > 0:
            avg_degree = (2 * num_edges) / num_nodes
            metrics.append(f"Average degree: {avg_degree:.2f}")

        metrics.append("\nNode Degrees:")
        for node_id, node in self.scene.nodes.items():
            degree = len(node.edges)
            metrics.append(f"{node_id}: {degree}")

        return "\n".join(metrics)
    
    def _restart_requested(self):
        self.close()

    

    
    



if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    while True:
        mode, ok = QInputDialog.getItem(None, "Select Mode", "Choose graph mode:", ["Homogeneous", "Heterogeneous"], 0, False)

        if not ok:
            break

        window = GraphEditor(graph_mode=mode)
        window.show()

        app.exec()

     
