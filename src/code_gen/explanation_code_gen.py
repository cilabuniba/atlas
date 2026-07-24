import networkx as nx
from torch_geometric.data import Data, HeteroData
from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.utils import to_networkx
import torch_geometric.datasets as data_pkg
import os
import torch
from .entire_dataset_code_gen import generate_node_type_styles


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
    for rel_type in explanation.edge_types:
        src_t, _, dst_t = rel_type
        if explanation[src_t].node_mask.ndim > 1:
            explanation[src_t].node_mask = explanation[src_t].node_mask.squeeze(1)
        if explanation[dst_t].node_mask.ndim > 1:
            explanation[dst_t].node_mask = explanation[dst_t].node_mask.squeeze(1)
        explanation[rel_type].edge_mask = (
            explanation[src_t].node_mask[data[rel_type].edge_index[0]].bool()
            & explanation[dst_t].node_mask[data[rel_type].edge_index[1]].bool()
            & explanation[rel_type].edge_mask.bool()
        )
    return explanation


def export_pyg_explanation_to_python(
    dataset: Data,
    explanation: Explanation,
    output_dir: str = "./",
    layout: str = "spring",
    scale: int = 1000,
) -> None:
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
            y = dataset.y[node].unsqueeze(0) if node < dataset.y.size(0) else dataset.y
            y = y.tolist() if y.size(0) > 1 or y.ndim > 1 else y.item()
            attrs = {
                "type": "Generic",
                "description": (f"Class: {y}" if hasattr(dataset, "y") else ""),
                "shape": "Circle",
                "pos": pos[node],
                "explained": (
                    hasattr(explanation, "node") and node == int(explanation.node)
                ),
            }

            f.write(f"G.add_node({repr(str(node))}, **{repr(attrs)})\n")

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
            f.write(f"    {repr(str(node))}: ({x}, {y}),\n")

        f.write("}\n\n")

        f.write("nx.draw(G, pos=pos, with_labels=True)\n")


def export_hetero_pyg_explanation_to_python(
    dataset: HeteroData,
    explanation: HeteroExplanation,
    output_dir: str = "./",
):
    """
    Export a PyG HeteroData explanation into a standalone NetworkX script
    compatible with the graph visualization widget.

    The exported graph contains only nodes and edges selected by the
    explanation hard masks.
    """
    explanation = preprocess_hetero_explanation(data=dataset, explanation=explanation)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/explanation_export.py"


    # ---------------------------------
    # Create temporary NetworkX graph
    # ---------------------------------

    G = nx.MultiDiGraph()


    node_styles = generate_node_type_styles(
        dataset.node_types
    )


    # ---------------------------------
    # Nodes from explanation mask
    # ---------------------------------

    node_type_iterator = dataset.node_types


    selected_nodes = {}


    for node_type in node_type_iterator:
        store = dataset[node_type]
        mask = explanation[node_type].node_mask

        # In case the mask is feature-level [N,D]
        if mask.dim() > 1:
            mask = mask.abs().mean(dim=-1)
        node_ids = torch.where(mask)[0].tolist()
        selected_nodes[node_type] = set(node_ids)

        for idx in node_ids:
            node_id = f"{node_type}:{idx}"
            description = node_id
            if hasattr(store, "y"):
                try:
                    description += (f" - Class: {int(store.y[idx])}")
                except:
                    pass

            G.add_node(
                node_id,
                type=node_type,
                description=description,
                shape=node_styles[node_type]["shape"],
                color=node_styles[node_type]["color"],
                explained=(
                    hasattr(explanation, "node")
                    and explanation.node == (node_type, idx)
                ),
            )

    # ---------------------------------
    # Edges from explanation mask
    # ---------------------------------

    edge_type_iterator = dataset.edge_types
    for edge_type in edge_type_iterator:
        if edge_type not in explanation.edge_types:
            continue
        src_type, relation, dst_type = edge_type
        edge_index = dataset[edge_type].edge_index
        edge_mask = explanation[edge_type].edge_mask
        edge_ids = torch.where(edge_mask)[0]

        for edge_id in edge_ids.tolist():
            src = edge_index[0, edge_id].item()
            dst = edge_index[1, edge_id].item()
            # Keep only edges between selected nodes
            if (
                src in selected_nodes[src_type]
                and
                dst in selected_nodes[dst_type]
            ):
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
        pos = nx.spring_layout(G,seed=42)
    else:
        pos = nx.random_layout(G)

    pos = {
        node: (
            float(x * 1000),
            float(y * 1000)
        )
        for node, (x, y) in pos.items()
    }

    # ---------------------------------
    # Write Python file
    # ---------------------------------
    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("import networkx as nx\n\n")
        f.write("G = nx.MultiDiGraph()\n\n")

        # Nodes
        for node, attrs in G.nodes(data=True):
            attrs["pos"] = pos[node]
            f.write(
                f"G.add_node("
                f"{repr(node)}, "
                f"**{repr(attrs)}"
                f")\n"
            )

        f.write("\n")

        # Edges
        for u, v, attrs in G.edges(data=True):
            f.write(
                f"G.add_edge("
                f"{repr(u)}, "
                f"{repr(v)}, "
                f"**{repr(attrs)}"
                f")\n"
            )

        f.write("\n")
        # Positions
        f.write(
            "pos = {\n"
        )
        for node, p in pos.items():
            f.write(
                f"    {repr(node)}: {p},\n"
            )
        f.write("}\n\n")

        f.write("nx.draw(G, pos=pos, with_labels=True)\n")
    print(f"Saved heterogeneous explanation graph to {output_file}")


def generate_explanation_code(parameters: dict) -> None:
    dataset_parameters = parameters["dataset"]
    dataset = get_dataset(**dataset_parameters)
    explanation_parameters = parameters["explanation"]
    explanation = load_explanation(**explanation_parameters)
    export_parameters = parameters["export"]
    return export_pyg_explanation_to_python(
        dataset=dataset, explanation=explanation, **export_parameters
    )


def generate_hetero_explanation_code(parameters: dict) -> None:
    dataset_parameters = parameters["dataset"]
    dataset = get_dataset(**dataset_parameters)
    explanation_parameters = parameters["explanation"]
    explanation = load_explanation(**explanation_parameters)
    export_parameters = parameters["export"]
    return export_hetero_pyg_explanation_to_python(
        dataset=dataset, explanation=explanation, **export_parameters
    )
