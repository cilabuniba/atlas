def export_networkx(scene):
    code = "import networkx as nx\n\n"
    code += "G = nx.Graph()\n\n"

    for node in scene.nodes.values():
        attrs = {
            "type": node.tipo,
            "description": node.attributes.get("description", ""),
            "shape": node.attributes.get("shape", "Circle"),
            "pos": (node.pos.x(), node.pos.y())
        }
        code += f"G.add_node('{node.id}', **{attrs})\n"

    added_edges = set()
    for node in scene.nodes.values():
        for edge in node.edges:
            edge_tuple = tuple(sorted([edge.source.id, edge.target.id]))
            if edge_tuple not in added_edges:
                attrs = {
                    "type": edge.tipo,
                    "description": edge.attributes.get("description", "")
                }
                code += f"G.add_edge('{edge.source.id}', '{edge.target.id}', **{attrs})\n"
                added_edges.add(edge_tuple)

    code += "\n# Optional: If you want to preserve the layout\n"
    code += "pos = {\n"
    for node in scene.nodes.values():
        code += f"    '{node.id}': ({node.pos.x()}, {node.pos.y()}),\n"
    code += "}\n"
    code += "nx.draw(G, pos=pos, with_labels=True)\n"
    return code


def export_igraph(scene):
    code = "import igraph as ig\n\n"
    code += "g = ig.Graph()\n\n"

    code += f"g.add_vertices({len(scene.nodes)})\n"
    code += "g.vs['name'] = " + str([str(node.id) for node in scene.nodes.values()]) + "\n"
    code += "g.vs['type'] = " + str([node.tipo for node in scene.nodes.values()]) + "\n"
    code += "g.vs['description'] = " + str([node.attributes.get("description", "") for node in scene.nodes.values()]) + "\n"
    code += "g.vs['shape'] = " + str([node.attributes.get("shape", "Circle") for node in scene.nodes.values()]) + "\n"

    edges = []
    added_edges = set()
    for node in scene.nodes.values():
        for edge in node.edges:
            edge_tuple = tuple(sorted([edge.source.id, edge.target.id]))
            if edge_tuple not in added_edges:
                edges.append((edge.source.id, edge.target.id))
                added_edges.add(edge_tuple)

    code += "edges = " + str(edges) + "\n"
    code += "g.add_edges(edges)\n\n"

    code += "# Layout positions\n"
    code += "layout = [\n"
    for node in scene.nodes.values():
        code += f"    ({node.pos.x()}, {node.pos.y()}),\n"
    code += "]\n"

    return code


def export_pyvis(scene):
    code = "from pyvis.network import Network\n\n"
    code += "net = Network()\n\n"

    for node in scene.nodes.values():
        code += f"net.add_node('{node.id}', x={node.pos.x()}, y={node.pos.y()}, title='{node.attributes.get('descrizione', '')}', label='{node.tipo}')\n"

    added_edges = set()
    for node in scene.nodes.values():
        for edge in node.edges:
            edge_tuple = tuple(sorted([edge.source.id, edge.target.id]))
            if edge_tuple not in added_edges:
                code += f"net.add_edge('{edge.source.id}', '{edge.target.id}', title='{edge.attributes.get('descrizione', '')}', label='{edge.tipo}')\n"
                added_edges.add(edge_tuple)

    code += "\nnet.show('graph.html')\n"
    return code


def export_graphtool(scene):
    code = "from graph_tool.all import *\n\n"
    code += "g = Graph()\n"
    code += "name = g.new_vertex_property('string')\n"
    code += "type = g.new_vertex_property('string')\n"
    code += "description = g.new_vertex_property('string')\n"
    code += "shape = g.new_vertex_property('string')\n"
    code += "pos = g.new_vertex_property('vector<double>')\n\n"

    code += "vertices = {}\n"
    for node in scene.nodes.values():
        code += "v = g.add_vertex()\n"
        code += f"name[v] = '{node.id}'\n"
        code += f"type[v] = '{node.tipo}'\n"
        code += f"description[v] = '{node.attributes.get('description', '')}'\n"
        code += f"shape[v] = '{node.attributes.get('shape', 'Circle')}'\n"
        code += f"pos[v] = [{node.pos.x()}, {node.pos.y()}]\n"
        code += f"vertices['{node.id}'] = v\n"

    added_edges = set()
    for node in scene.nodes.values():
        for edge in node.edges:
            edge_tuple = tuple(sorted([edge.source.id, edge.target.id]))
            if edge_tuple not in added_edges:
                code += f"g.add_edge(vertices['{edge.source.id}'], vertices['{edge.target.id}'])\n"
                added_edges.add(edge_tuple)

    return code


def export_dgl(scene):
    code = "import dgl\nimport torch\n\n"


    node_type_groups = {}
    for node in scene.nodes.values():
        tipo = node.tipo or "Generic"
        if tipo not in node_type_groups:
            node_type_groups[tipo] = []
        node_type_groups[tipo].append(node)

 
    local_node_ids = {}
    for tipo, nodes in node_type_groups.items():
        for i, node in enumerate(nodes):
            local_node_ids[node.id] = (tipo, i)

   
    edge_type_map = {}
    for node in scene.nodes.values():
        for edge in node.edges:
            if edge.source.id != node.id:
                continue  
            src_type, src_idx = local_node_ids[edge.source.id]
            dst_type, dst_idx = local_node_ids[edge.target.id]
            etype = edge.tipo or "relates_to"
            key = (src_type, etype, dst_type)
            if key not in edge_type_map:
                edge_type_map[key] = ([], [])
            edge_type_map[key][0].append(src_idx)
            edge_type_map[key][1].append(dst_idx)

   
    
    for tipo, nodes in node_type_groups.items():
        code += f"# {tipo}: {[node.id for node in nodes]}\n"

    
    code += "data_dict = {\n"
    for (srctype, etype, dsttype), (srcs, dsts) in edge_type_map.items():
        code += f"    ('{srctype}', '{etype}', '{dsttype}'): (torch.tensor({srcs}), torch.tensor({dsts})),\n"
    code += "}\n\n"


    code += "num_nodes_dict = {\n"
    for tipo, nodes in node_type_groups.items():
        code += f"    '{tipo}': {len(nodes)},\n"
    code += "}\n\n"

    code += "g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)\n"
    return code




def export_snap(scene):
    code = "import snap\n\n"
    code += "G = snap.TNGraph.New()\n\n"

    code += "# Mapping ID → int index\n"
    code += "node_mapping = {\n"
    for i, node in enumerate(scene.nodes.values()):
        code += f"    '{node.id}': {i},  # type: {node.tipo}, description: {node.attributes.get('description', '')}\n"
    code += "}\n\n"

    for node in scene.nodes.values():
        code += f"G.AddNode(node_mapping['{node.id}'])\n"

    added_edges = set()
    for node in scene.nodes.values():
        for edge in node.edges:
            edge_tuple = tuple(sorted([edge.source.id, edge.target.id]))
            if edge_tuple not in added_edges:
                code += f"G.AddEdge(node_mapping['{edge.source.id}'], node_mapping['{edge.target.id}'])\n"
                added_edges.add(edge_tuple)

    return code


def export_pygraphviz(scene):
    code = "import pygraphviz as pgv\n\n"
    code += "G = pgv.AGraph(strict=False, directed=False)\n\n"

    for node in scene.nodes.values():
        pos = f"{node.pos.x()},{node.pos.y()}!"
        code += f"G.add_node('{node.id}', pos='{pos}', type='{node.tipo}', description='{node.attributes.get('description', '')}', shape='{node.attributes.get('shape', 'Circle')}')\n"

    added_edges = set()
    for node in scene.nodes.values():
        for edge in node.edges:
            edge_tuple = tuple(sorted([edge.source.id, edge.target.id]))
            if edge_tuple not in added_edges:
                code += f"G.add_edge('{edge.source.id}', '{edge.target.id}', type='{edge.tipo}', description='{edge.attributes.get('description', '')}')\n"
                added_edges.add(edge_tuple)

    return code

def export_pyg(scene):
    code = "import torch\n"
    code += "from torch_geometric.data import HeteroData\n\n"
    code += "data = HeteroData()\n\n"

    # Raggruppa nodi per tipo
    type_to_nodes = {}
    node_to_index = {}
    for node in scene.nodes.values():
        type_to_nodes.setdefault(node.tipo, []).append(node)

    # Definisci nodi con feature e attributi
    for ntype, nodes in type_to_nodes.items():
        code += f"# Nodes of type: {ntype}\n"
        code += f"data['{ntype}'].x = torch.eye({len(nodes)})\n"
        labels = [f"'{n.id} - {n.attributes.get('description', '')}'" for n in nodes]
        code += f"data['{ntype}'].name = [{', '.join(labels)}]\n\n"
        for idx, n in enumerate(nodes):
            node_to_index[n.id] = (ntype, idx)

    # Definisci edge_index per tipo di relazione
    edge_map = {}
    for node in scene.nodes.values():
        for edge in node.edges:
            src_id = edge.source.id
            dst_id = edge.target.id
            rel_type = edge.tipo or "relates_to"
            src_type, src_idx = node_to_index[src_id]
            dst_type, dst_idx = node_to_index[dst_id]

            key = (src_type, rel_type, dst_type)
            edge_map.setdefault(key, [[], []])
            edge_map[key][0].append(src_idx)
            edge_map[key][1].append(dst_idx)

    for (src_t, rel, dst_t), (srcs, dsts) in edge_map.items():
        code += f"data[('{src_t}', '{rel}', '{dst_t}')].edge_index = torch.tensor([\n"
        code += f"    {srcs},\n    {dsts}\n], dtype=torch.long)\n\n"

    return code

