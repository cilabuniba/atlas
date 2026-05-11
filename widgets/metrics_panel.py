from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QScrollArea
import networkx as nx

class MetricsPanel(QWidget):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        
        # Create container widget for scroll area
        container = QWidget()
        scroll.setWidget(container)
        self.layout = QVBoxLayout(container)
        
        # Basic metrics
        self.nodes_label = QLabel("Nodes: 0")
        self.edges_label = QLabel("Edges: 0")
        self.density_label = QLabel("Density: 0")
        self.avg_degree_label = QLabel("Average Degree: 0")
        
        # Advanced metrics
        self.clustering_label = QLabel("Average Clustering: N/A")
        self.diameter_label = QLabel("Diameter: N/A")
        self.radius_label = QLabel("Radius: N/A")
        self.center_nodes_label = QLabel("Center Nodes: N/A")
        self.periphery_nodes_label = QLabel("Periphery Nodes: N/A")
        
        # Centrality metrics
        self.degree_centrality_label = QLabel("Degree Centrality:")
        self.betweenness_centrality_label = QLabel("Betweenness Centrality:")
        self.closeness_centrality_label = QLabel("Closeness Centrality:")
        self.eigenvector_centrality_label = QLabel("Eigenvector Centrality:")
        
        # Add all labels to layout
        self.layout.addWidget(QLabel("Basic Metrics:"))
        self.layout.addWidget(self.nodes_label)
        self.layout.addWidget(self.edges_label)
        self.layout.addWidget(self.density_label)
        self.layout.addWidget(self.avg_degree_label)
        
        self.layout.addWidget(QLabel("\nAdvanced Metrics:"))
        self.layout.addWidget(self.clustering_label)
        self.layout.addWidget(self.diameter_label)
        self.layout.addWidget(self.radius_label)
        self.layout.addWidget(self.center_nodes_label)
        self.layout.addWidget(self.periphery_nodes_label)
        
        self.layout.addWidget(QLabel("\nCentrality Metrics:"))
        self.layout.addWidget(self.degree_centrality_label)
        self.layout.addWidget(self.betweenness_centrality_label)
        self.layout.addWidget(self.closeness_centrality_label)
        self.layout.addWidget(self.eigenvector_centrality_label)
        
        # Node degrees
        self.layout.addWidget(QLabel("\nNode Degrees:"))
        self.degrees_text = QTextEdit()
        self.degrees_text.setReadOnly(True)
        self.degrees_text.setMaximumHeight(200)
        self.degrees_text.setMinimumHeight(100)
        self.layout.addWidget(self.degrees_text)
        self.layout.addStretch()
        
        
    def update_metrics(self):
        num_nodes = len(self.scene.nodes)
        num_edges = sum(len(node.edges) for node in self.scene.nodes.values()) // 2
        
        # Update basic metrics
        self.nodes_label.setText(f"Nodes: {num_nodes}")
        self.edges_label.setText(f"Edges: {num_edges}")
        
        if num_nodes > 1:
            density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
            self.density_label.setText(f"Density: {density:.3f}")
        else:
            self.density_label.setText("Density: N/A")
        
        if num_nodes > 0:
            avg_degree = (2 * num_edges) / num_nodes
            self.avg_degree_label.setText(f"Average Degree: {avg_degree:.2f}")
            
            # Create NetworkX graph for advanced metrics
            G = nx.Graph()
            
            # Add nodes and edges
            for node in self.scene.nodes.values():
                G.add_node(node.id)
            
            added_edges = set()
            for node in self.scene.nodes.values():
                for edge in node.edges:
                    edge_tuple = tuple(sorted([edge.source.id, edge.target.id]))
                    if edge_tuple not in added_edges:
                        G.add_edge(edge.source.id, edge.target.id)
                        added_edges.add(edge_tuple)
            
            try:
                # Calculate advanced metrics
                clustering = nx.average_clustering(G)
                self.clustering_label.setText(f"Average Clustering: {clustering:.3f}")
                
                if nx.is_connected(G):
                    diameter = nx.diameter(G)
                    radius = nx.radius(G)
                    center = nx.center(G)
                    periphery = nx.periphery(G)
                    
                    self.diameter_label.setText(f"Diameter: {diameter}")
                    self.radius_label.setText(f"Radius: {radius}")
                    self.center_nodes_label.setText(f"Center Nodes: {', '.join(center)}")
                    self.periphery_nodes_label.setText(f"Periphery Nodes: {', '.join(periphery)}")
                else:
                    self.diameter_label.setText("Diameter: N/A (Graph not connected)")
                    self.radius_label.setText("Radius: N/A (Graph not connected)")
                    self.center_nodes_label.setText("Center Nodes: N/A (Graph not connected)")
                    self.periphery_nodes_label.setText("Periphery Nodes: N/A (Graph not connected)")
                
                # Calculate centrality metrics
                degree_cent = nx.degree_centrality(G)
                betweenness_cent = nx.betweenness_centrality(G)
                closeness_cent = nx.closeness_centrality(G)
                try:
                    eigenvector_cent = nx.eigenvector_centrality(G)
                except:
                    eigenvector_cent = {node: 0 for node in G.nodes()}
                
                # Format centrality metrics
                self.degree_centrality_label.setText("Degree Centrality:\n" + 
                    "\n".join(f"{node}: {cent:.3f}" for node, cent in degree_cent.items()))
                self.betweenness_centrality_label.setText("Betweenness Centrality:\n" + 
                    "\n".join(f"{node}: {cent:.3f}" for node, cent in betweenness_cent.items()))
                self.closeness_centrality_label.setText("Closeness Centrality:\n" + 
                    "\n".join(f"{node}: {cent:.3f}" for node, cent in closeness_cent.items()))
                self.eigenvector_centrality_label.setText("Eigenvector Centrality:\n" + 
                    "\n".join(f"{node}: {cent:.3f}" for node, cent in eigenvector_cent.items()))
                
            except Exception as e:
                print(f"Error calculating advanced metrics: {e}")
            
            # Update node degrees text
            degrees_text = ""
            for node_id, node in self.scene.nodes.items():
                degree = len(node.edges)
                degrees_text += f"{node_id}: {degree}\n"
            self.degrees_text.setText(degrees_text)
        else:
            self.avg_degree_label.setText("Average Degree: N/A")
            self.clustering_label.setText("Average Clustering: N/A")
            self.diameter_label.setText("Diameter: N/A")
            self.radius_label.setText("Radius: N/A")
            self.center_nodes_label.setText("Center Nodes: N/A")
            self.periphery_nodes_label.setText("Periphery Nodes: N/A")
            self.degree_centrality_label.setText("Degree Centrality: N/A")
            self.betweenness_centrality_label.setText("Betweenness Centrality: N/A")
            self.closeness_centrality_label.setText("Closeness Centrality: N/A")
            self.eigenvector_centrality_label.setText("Eigenvector Centrality: N/A")
            self.degrees_text.setText("") 