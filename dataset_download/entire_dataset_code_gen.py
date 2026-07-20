from pathlib import Path
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch_geometric.datasets as data_class_dict
from tqdm import tqdm
import os


def download_dataset(data_class, dataset_cfg: dict = {}) -> Data:
    return data_class_dict.__dict__[data_class](**dataset_cfg)[0]


def export_pyg_graph_to_python(
    data: Data,
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

    # Convert PyG graph to NetworkX
    G = to_networkx(
        data,
        to_undirected=True,
        remove_self_loops=True,
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

            attrs = {
                "type": "Generic",
                "description": (
                    f"Class: {int(data.y[node])}" if hasattr(data, "y") else ""
                ),
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


def generate_dataset_code(parameters: dict) -> None:
    dataset_parameters = parameters["dataset"]
    exporter_parameters = parameters["export"]

    dataset = download_dataset(**dataset_parameters)
    return export_pyg_graph_to_python(data=dataset, **exporter_parameters)
