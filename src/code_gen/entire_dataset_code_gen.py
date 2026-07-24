from pathlib import Path
import networkx as nx
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx, subgraph
import torch_geometric.datasets as data_class_dict
from tqdm import tqdm
import os
import torch


def download_dataset(data_class, dataset_cfg: dict = {}) -> Data:
    return data_class_dict.__dict__[data_class](**dataset_cfg)[0]


def filter_graph(data: Data, num_nodes: int):
    if num_nodes > data.x.size(0):
        return data
    x = data.x[:num_nodes]
    edge_index, _ = subgraph(
        torch.arange(num_nodes), data.edge_index, relabel_nodes=True
    )
    y = data.y[:num_nodes]
    return Data(x=x, edge_index=edge_index, y=y)


def export_pyg_graph_to_python(
    data: Data,
    num_nodes: int = None,
    output_dir: str = "./",
    layout: str = "spring",
    scale: int = 1000,
    use_tqdm: bool = False,
):
    """
    Export a homogeneous PyTorch Geometric graph into a standalone Python
    script that recreates the graph using NetworkX.

    The generated file has the form:

        import networkx as nx

        G = nx.Graph()
        G.add_node(...)
        ...
        G.add_edge(...)
        ...

        pos = {...}
        nx.draw(G, pos=pos, with_labels=True)
    """

    if num_nodes is not None:
        data = filter_graph(data=data, num_nodes=num_nodes)

    # Convert PyG graph to NetworkX
    G = to_networkx(
        data,
        to_undirected=True,
        remove_self_loops=False,
    )

    # Compute node positions
    if layout == "spring":
        pos = nx.spring_layout(G, iterations=5)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    else:
        raise ValueError(f"Unknown layout '{layout}'")

    # Scale positions
    pos = {
        node: (
            round(float(x * scale), 3),
            round(float(y * scale), 3),
        )
        for node, (x, y) in pos.items()
    }

    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(f"{output_dir}/graph_export.py")
    with open(output_file, "w+", encoding="utf-8") as f:

        f.write("import networkx as nx\n\n")
        f.write("G = nx.Graph()\n\n")

        print("Start nodes!")
        # Nodes
        node_iterator = tqdm(G.nodes(), desc="Nodes") if use_tqdm else G.nodes()
        for node in node_iterator:
            y = data.y[node].unsqueeze(0) if node < data.y.size(0) else data.y
            y = y.tolist() if y.size(0) > 1 or y.ndim > 1 else y.item()
            attrs = {
                "type": "Generic",
                "description": (f"Class: {y}" if hasattr(data, "y") else ""),
                "shape": "Circle",
                "pos": pos[node],
            }

            f.write(f"G.add_node({repr(str(node))}, **{repr(attrs)})\n")

        f.write("\n")

        # Edges
        edge_iterator = tqdm(G.edges(), desc="Edges") if use_tqdm else G.edges()
        for u, v in edge_iterator:

            attrs = {
                "type": "Edge",
                "description": "",
            }

            f.write(f"G.add_edge({repr(str(u))}, {repr(str(v))}, **{repr(attrs)})\n")

        f.write("\n")

        # Position dictionary
        f.write("pos = {\n")
        for node in G.nodes():
            f.write(f"    {repr(str(node))}: ({pos[node][0]}, {pos[node][1]}),\n")
        f.write("}\n\n")

        f.write("nx.draw(G, pos=pos, with_labels=True)\n")

    print(f"Saved graph to {output_file.resolve()}")


def filter_hetero_graph(data: HeteroData, num_nodes: int):
    """
    Keep only the first num_nodes nodes of each node type and
    remove invalid edges.
    """

    data = data.clone()

    kept_nodes = {}

    # -----------------------------
    # Filter nodes
    # -----------------------------
    for node_type in data.node_types:
        store = data[node_type]
        old_num_nodes = store.num_nodes
        keep = min(num_nodes, old_num_nodes)
        kept_nodes[node_type] = keep

        for key, value in list(store.items()):
            if isinstance(value, torch.Tensor):
                if value.size(0) == old_num_nodes:
                    store[key] = value[:keep]

            elif isinstance(value, list):
                if len(value) == old_num_nodes:
                    store[key] = value[:keep]

        store.num_nodes = keep

    # -----------------------------
    # Filter edges
    # -----------------------------
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        store = data[edge_type]
        edge_index = store.edge_index
        mask = (edge_index[0] < kept_nodes[src_type]) & (
            edge_index[1] < kept_nodes[dst_type]
        )
        store.edge_index = edge_index[:, mask]
        num_edges = edge_index.size(1)
        for key, value in list(store.items()):
            if key == "edge_index":
                continue

            if isinstance(value, torch.Tensor):
                if value.size(0) == num_edges:
                    store[key] = value[mask]

    return data


def generate_node_type_styles(node_types):
    """
    Dynamically assign colors and shapes to node types.
    """

    shapes = [
        "Circle",
        "Rectangle",
        "Ellipse",
        "Diamond",
    ]

    # Color palette (hex)
    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    ]

    styles = {}

    for i, node_type in enumerate(node_types):

        styles[node_type] = {
            "shape": shapes[i % len(shapes)],
            "color": colors[i % len(colors)],
        }

    return styles


def export_pyg_hetero_graph_to_python(
    data: HeteroData,
    num_nodes: int = None,
    output_dir: str = "./",
    use_tqdm: bool = False,
):
    """
    Export PyG HeteroData into a standalone NetworkX script
    compatible with the graph visualization widget.
    """

    if num_nodes is not None:
        data = filter_hetero_graph(data, num_nodes)

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/graph_export.py"

    # ---------------------------------
    # Create temporary NetworkX graph
    # ---------------------------------
    G = nx.MultiDiGraph()
    # Nodes

    node_styles = generate_node_type_styles(data.node_types)

    node_type_iterator = tqdm(data.node_types) if use_tqdm else data.node_types
    for node_type in node_type_iterator:
        store = data[node_type]
        for idx in range(store.num_nodes):
            node_id = f"{node_type}:{idx}"
            description = node_id
            if hasattr(store, "y"):
                try:
                    description += f" - Class: {int(store.y[idx])}"
                except:
                    pass
            G.add_node(
                node_id,
                type=node_type,
                description=description,
                shape=node_styles[node_type]["shape"],
                color=node_styles[node_type]["color"],
            )

    # Edges
    edge_type_iterator = tqdm(data.edge_types) if use_tqdm else data.edge_types
    for edge_type in edge_type_iterator:
        src_type, relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        for src, dst in edge_index.t().tolist():
            G.add_edge(
                f"{src_type}:{src}",
                f"{dst_type}:{dst}",
                type=relation,
                description=relation,
            )

    # ---------------------------------
    # Compute layout
    # ---------------------------------
    if len(G) < 5000:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.random_layout(G)

    pos = {node: (float(x * 1000), float(y * 1000)) for node, (x, y) in pos.items()}

    # ---------------------------------
    # Write Python file
    # ---------------------------------

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("import networkx as nx\n\n")

        f.write("G = nx.MultiDiGraph()\n\n")

        # Nodes
        for node, attrs in G.nodes(data=True):
            attrs["pos"] = pos[node]
            f.write(f"G.add_node(" f"{repr(node)}, " f"**{repr(attrs)}" f")\n")
        f.write("\n")

        # Edges
        for u, v, attrs in G.edges(data=True):
            f.write(
                f"G.add_edge(" f"{repr(u)}, " f"{repr(v)}, " f"**{repr(attrs)}" f")\n"
            )

        f.write("\n")

        # Positions
        f.write("pos = {\n")

        for node, p in pos.items():
            f.write(f"    {repr(node)}: {p},\n")

        f.write("}\n\n")

        f.write("nx.draw(G, pos=pos, with_labels=True)\n")

    print(f"Saved heterogeneous graph to {output_file}")


def generate_dataset_code(parameters: dict) -> None:
    dataset_parameters = parameters["dataset"]
    exporter_parameters = parameters["export"]

    dataset = download_dataset(**dataset_parameters)
    return export_pyg_graph_to_python(data=dataset, **exporter_parameters)


def generate_hetero_dataset_code(parameters: dict) -> None:
    dataset_parameters = parameters["dataset"]
    exporter_parameters = parameters["export"]
    dataset = download_dataset(**dataset_parameters)
    return export_pyg_hetero_graph_to_python(data=dataset, **exporter_parameters)
