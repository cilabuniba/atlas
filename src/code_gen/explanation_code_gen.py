import networkx as nx
from torch_geometric.data import Data, HeteroData
from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.utils import to_networkx
import torch_geometric.datasets as data_pkg
import os
import torch


def get_dataset(
    data_class: str,
    dataset_cfg: dict = {},
    idx: int = 0,
    custom: bool = False,
) -> Data | HeteroData:
    return (
        data_pkg.__dict__[data_class](**dataset_cfg)[idx]
        if not custom
        else load_test_set(data_class)
    )


def load_test_set(fname: str) -> Data | HeteroData:
    return torch.load(fname, map_location="cpu", weights_only=False)


def load_explanation(fname: str) -> Explanation | HeteroData:
    return torch.load(fname, map_location="cpu", weights_only=False)


def preprocess_explanation(data: Data, explanation: Explanation) -> Explanation:
    explanation.node_mask = explanation.node_mask.squeeze(1)
    explanation.edge_mask = (
        explanation.node_mask[data.edge_index[0]].bool()
        & explanation.node_mask[data.edge_index[1]].bool()
        & explanation.edge_mask.bool()
    )
    return explanation


def preprocess_hetero_explanation(
    data: HeteroData, explanation: HeteroExplanation
) -> HeteroExplanation:
    pass


def export_pyg_explanation_to_python(dataset: Data, explanation: Explanation, output_dir: str = "./", layout: str = "spring", scale: int =1000,) -> None:
    explanation = preprocess_explanation(data=dataset, explanation=explanation)

    G = to_networkx(
        dataset,
        to_undirected=True,
        remove_self_loops=False,
    )

    # Compute the layout ONCE on the complete graph
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

    pos = {
        node: (
            round(float(x * scale), 3),
            round(float(y * scale), 3),
        )
        for node, (x, y) in pos.items()
    }

    selected_nodes = set(torch.where(explanation.node_mask)[0].tolist())
    selected_edges = torch.where(explanation.edge_mask)[0]

    edge_index = dataset.edge_index[:, selected_edges]

    output_file = f"{output_dir}/explanation_export.py"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("import networkx as nx\n\n")
        f.write("G = nx.Graph()\n\n")

        # -------------------------
        # Nodes
        # -------------------------

        for node in selected_nodes:

            attrs = {
                "type": "Generic",
                "description": f"Class: {int(dataset.y[node])}",
                "shape": "Circle",
                "pos": pos[node],
                "explained": (
                    hasattr(explanation, "node")
                    and node == int(explanation.node)
                ),
            }

            f.write(
                f"G.add_node({repr(str(node))}, **{repr(attrs)})\n"
            )

        f.write("\n")

        # -------------------------
        # Edges
        # -------------------------

        for src, dst in edge_index.t().tolist():

            if src in selected_nodes and dst in selected_nodes:

                attrs = {
                    "type": "Edge",
                    "description": "",
                }

                f.write(
                    f"G.add_edge({repr(str(src))}, {repr(str(dst))}, **{repr(attrs)})\n"
                )

        f.write("\n")

        f.write("pos = {\n")

        for node in selected_nodes:
            x, y = pos[node]
            f.write(
                f"    {repr(str(node))}: ({x}, {y}),\n"
            )

        f.write("}\n\n")

        f.write("nx.draw(G, pos=pos, with_labels=True)\n")



def generate_explanation_code(parameters: dict) -> None:
    dataset_parameters = parameters["dataset"]
    dataset = get_dataset(**dataset_parameters)
    explanation_parameters = parameters["explanation"]
    explanation = load_explanation(**explanation_parameters)


    return export_pyg_explanation_to_python(dataset=dataset, explanation=explanation)