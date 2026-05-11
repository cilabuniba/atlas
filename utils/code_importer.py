import ast, re
from PyQt6.QtCore import QPointF

AVAILABLE_LIBRARIES = {}

try:
    import networkx as nx
    AVAILABLE_LIBRARIES['networkx'] = nx
except ImportError:
    pass

try:
    import graph_tool.all as gt
    AVAILABLE_LIBRARIES['graph-tool'] = gt
except ImportError:
    pass

try:
    import igraph as ig
    AVAILABLE_LIBRARIES['igraph'] = ig
except ImportError:
    pass

try:
    from pyvis.network import Network
    AVAILABLE_LIBRARIES['pyvis'] = Network
except ImportError:
    pass

try:
    import pygraphviz as pgv
    AVAILABLE_LIBRARIES['pygraphviz'] = pgv
except ImportError:
    pass

try:
    import dgl
    AVAILABLE_LIBRARIES['dgl'] = dgl
except ImportError:
    pass

try:
    import snap
    AVAILABLE_LIBRARIES['snap'] = snap
except ImportError:
    pass

class GraphImporter:
    @staticmethod
    def import_from_code(code, scene):
        """Import a graph from Python code into the scene"""
        try:
            tree = ast.parse(code)
            scene.clear_all()

            if "networkx" in code:
                if 'networkx' not in AVAILABLE_LIBRARIES:
                    raise ImportError("NetworkX is not installed. Please install it with 'pip install networkx'")
                GraphImporter._import_networkx(code, scene)
            elif "graph_tool" in code or "graph-tool" in code:
                if 'graph-tool' not in AVAILABLE_LIBRARIES:
                    raise ImportError("graph-tool is not installed. Please install it with your system package manager or conda")
                GraphImporter._import_graphtool(code, scene)
            elif "igraph" in code:
                if 'igraph' not in AVAILABLE_LIBRARIES:
                    raise ImportError("igraph is not installed. Please install it with 'pip install python-igraph'")
                GraphImporter._import_igraph(code, scene)
            elif "pyvis" in code:
                if 'pyvis' not in AVAILABLE_LIBRARIES:
                    raise ImportError("PyVis is not installed. Please install it with 'pip install pyvis'")
                GraphImporter._import_pyvis(code, scene)
            elif "pygraphviz" in code:
                if 'pygraphviz' not in AVAILABLE_LIBRARIES:
                    raise ImportError("PyGraphviz is not installed. Please install it with 'pip install pygraphviz'")
                GraphImporter._import_pygraphviz(code, scene)
            else:
                raise ValueError("No supported graph library found in code. Supported libraries: " + ", ".join(AVAILABLE_LIBRARIES.keys()))
            
            return True
        except Exception as e:
            raise ValueError(f"Error importing graph: {str(e)}")
    
    @staticmethod
    def _import_networkx(code, scene):
        """Import from NetworkX code"""
        if 'networkx' not in AVAILABLE_LIBRARIES:
            raise ImportError("NetworkX is not installed")
            
        nx = AVAILABLE_LIBRARIES['networkx']    
        G = nx.Graph()
        
        globals_dict = {
            "nx": AVAILABLE_LIBRARIES['networkx'],
            "networkx": AVAILABLE_LIBRARIES['networkx'],
            "G": G
        }
        locals_dict = {}
        
        
        try:
            import re
            code = re.sub(r'nx\.draw\([^\)]*\)', '', code)

            exec(code, globals_dict, locals_dict)
        except Exception as e:
            raise ValueError(f"Error executing code: {str(e)}")
        
        G = locals_dict.get("G", G)
        pos = locals_dict.get("pos")
        
        if pos is None:
            pos = nx.spring_layout(G)
        
        node_map = {}

        for node_id, attrs in G.nodes(data=True):
            try:
              
                pos_tuple = attrs.get("pos")
                if not pos_tuple:
                    raise ValueError(f"Node {node_id} is missing 'pos' attribute.")

                x, y = pos_tuple
                scene_pos = QPointF(float(x), float(y))

             
                tipo = attrs.get("type", "Generic")
                descrizione = attrs.get("description", "")
                forma = attrs.get("form", "Circle")

                node_attrs = {
                    "description": descrizione,
                    "shape": forma
                }

                scene_node = scene.add_node(node_id, scene_pos, tipo=tipo, attributes=node_attrs)
                node_map[node_id] = scene_node

            except Exception as e:
                raise ValueError(f"Error adding node '{node_id}': {str(e)}")

        
        # Add edges
        for source, target, attrs in G.edges(data=True):
            try:
                if source in node_map and target in node_map:
                    tipo = attrs.get("type", "Relation")
                    descrizione = attrs.get("description", "")
                    edge_attrs = {"description": descrizione}

                    scene.add_edge(node_map[source], node_map[target], tipo=tipo, attributes=edge_attrs)
            except Exception as e:
                raise ValueError(f"Error adding edge {source} → {target}: {str(e)}")

                        
        return True
    
    @staticmethod
    def _import_graphtool(code, scene):
        """Import from graph-tool code"""
        try:
            g = gt.Graph()
            name = g.new_vertex_property('string')
            pos = g.new_vertex_property('vector<double>')
            vertices = {}
            
            globals_dict = {
                "graph_tool": AVAILABLE_LIBRARIES['graph-tool'],
                "gt": AVAILABLE_LIBRARIES['graph-tool'],
                "Graph": gt.Graph,
                "vector": gt.vector
            }
            locals_dict = {
                "g": g,
                "name": name,
                "pos": pos,
                "vertices": vertices
            }
            
            exec(code, globals_dict, locals_dict)
            
            g = locals_dict.get("g", g)
            pos = locals_dict.get("pos", pos)
            
            vertex_list = []
            for v in g.vertices():
                try:
                    x, y = pos[v][0], pos[v][1]
                    scene_pos = QPointF(float(x), float(y))
                    node = scene.add_node(scene_pos)
                    vertex_list.append(node)
                except Exception as e:
                    raise ValueError(f"Error processing vertex {v}: {str(e)}")
            
            for e in g.edges():
                try:
                    source_idx = int(e.source())
                    target_idx = int(e.target())
                    scene.add_edge(vertex_list[source_idx], vertex_list[target_idx])
                except Exception as e:
                    raise ValueError(f"Error processing edge {e}: {str(e)}")
                
        except Exception as e:
            raise ValueError(f"Error importing graph-tool graph: {str(e)}")
    
    @staticmethod
    def _import_igraph(code, scene):
        """Import from igraph code with manual node positioning support"""
        if 'igraph' not in AVAILABLE_LIBRARIES:
            raise ImportError("igraph is not installed. Please install it with 'pip install python-igraph'")
        
        ig = AVAILABLE_LIBRARIES['igraph']
        
        globals_dict = {
            "igraph": ig,
            "ig": ig,
            "Graph": ig.Graph
        }
        locals_dict = {}

        try:
            exec(code, globals_dict, locals_dict)
        except Exception as e:
            raise ValueError(f"Error executing code: {str(e)}")
        
        graph = None
        for var in locals_dict.values():
            if isinstance(var, ig.Graph):
                graph = var
                break
        
        if graph is None:
            raise ValueError("No igraph Graph found in the code.")
        
        if graph.vcount() == 0:
            raise ValueError("The graph has no nodes.")
        
        layout = locals_dict.get('layout')

        if layout is not None:
            if isinstance(layout, list) and all(isinstance(pos, (list, tuple)) and len(pos) >= 2 for pos in layout):
                node_positions = [QPointF(float(x), float(y)) for x, y in layout]
            elif isinstance(layout, dict):
                node_positions = [QPointF(float(*layout.get(idx, (0, 0)))) for idx in range(graph.vcount())]
            elif isinstance(layout, ig.Layout):
                node_positions = [QPointF(float(pos[0]), float(pos[1])) for pos in layout]
            else:
                raise ValueError("The 'layout' variable must be an igraph Layout object, a list of tuples, or a dictionary.")
        else:
            try:
                layout = graph.layout('auto')
                node_positions = [QPointF(float(pos[0]), float(pos[1])) for pos in layout]
            except Exception as e:
                raise ValueError(f"Failed to generate a default layout: {str(e)}")

        if len(node_positions) != graph.vcount():
            raise ValueError("Layout dimension does not match the number of vertices in the graph.")

        node_map = {}
        for idx in range(graph.vcount()):
            pos = node_positions[idx]
            scene_node = scene.add_node(pos)
            node_map[idx] = scene_node

        for edge in graph.es:
            source = edge.source
            target = edge.target
            if source in node_map and target in node_map:
                scene.add_edge(node_map[source], node_map[target])
            else:
                raise ValueError(f"Edge references invalid vertices: {source} -> {target}")

        return True

    @staticmethod
    def _import_pyvis(code, scene):
        """Import from PyVis code"""
        if 'pyvis' not in AVAILABLE_LIBRARIES:
            raise ImportError("pyvis is not installed. Please install it with 'pip install pyvis'")

        from pyvis.network import Network

        globals_dict = {
            "Network": Network,
            "net": None
        }
        locals_dict = {}

        code = re.sub(r"net\.show\([^\)]*\)", "", code)

        try:
            exec(code, globals_dict, locals_dict)
        except Exception as e:
            raise ValueError(f"Error executing code: {str(e)}")

        net = None
        for var in locals_dict.values():
            if isinstance(var, Network):
                net = var
                break

        if net is None:
            raise ValueError("No PyVis Network object found in the code.")

        if not net.nodes:
            raise ValueError("The graph has no nodes.")

        node_positions = {}
        for node in net.nodes:
            node_id = str(node["id"])
            x = node.get("x", None)
            y = node.get("y", None)
            
            if x is None or y is None:
                raise ValueError(f"Node {node_id} is missing position data (x, y). Please export positions.")

            node_positions[node_id] = QPointF(float(x), float(y))

        node_map = {}
        for node in net.nodes:
            node_id = str(node["id"])
            pos = node_positions[node_id]
            scene_node = scene.add_node(pos)
            node_map[node_id] = scene_node

        for edge in net.edges:
            source = str(edge["from"])
            target = str(edge["to"])
            if source in node_map and target in node_map:
                scene.add_edge(node_map[source], node_map[target])
            else:
                raise ValueError(f"Edge references invalid vertices: {source} -> {target}")

        return True


    @staticmethod
    def _import_pygraphviz(code, scene):
        """Import from PyGraphviz code"""
        if 'pygraphviz' not in AVAILABLE_LIBRARIES:
            raise ImportError("PyGraphviz is not installed")
            
        G = AVAILABLE_LIBRARIES['pygraphviz'].AGraph(strict=False, directed=False)
        
        globals_dict = {
            "pgv": AVAILABLE_LIBRARIES['pygraphviz'],
            "pygraphviz": AVAILABLE_LIBRARIES['pygraphviz'],
            "G": G
        }
        locals_dict = {}
        
        try:
            exec(code, globals_dict, locals_dict)
        except Exception as e:
            raise ValueError(f"Error executing code: {str(e)}")
        
        G = locals_dict.get("G", G)
        
        node_map = {}
        for node in G.nodes():
            try:
                pos_str = G.get_node(node).attr.get("pos", "")
                if pos_str:
                    x, y = map(float, pos_str.strip("!").split(","))
                    scene_pos = QPointF(x, y)
                else:
                    raise ValueError(f"Node {node} is missing 'pos' attribute.")
                
                data = G.get_node(node).attr
                tipo = data.get("forma", "Generico")
                descrizione = data.get("descrizione", "")
                attributes = {"descrizione": descrizione, "forma": tipo}
                
                scene_node = scene.add_node(node, scene_pos, tipo=tipo, attributes=attributes)
                node_map[node] = scene_node

            except Exception as e:
                raise ValueError(f"Error adding node '{node}': {str(e)}")

        
        for edge in G.edges(data=True):
            try:
                source, target, data = edge
                if source in node_map and target in node_map:
                    tipo = data.get("tipo", "Collega")
                    descrizione = data.get("descrizione", "")
                    attributes = {"descrizione": descrizione}
                    scene.add_edge(node_map[source], node_map[target], tipo=tipo, attributes=attributes)

            except Exception as e:
                raise ValueError(f"Error adding edge {edge}: {str(e)}")
                    
        return True

