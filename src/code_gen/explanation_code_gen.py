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


def importance_to_hex(importance: float) -> str:
    """
    Map an importance score in [0, 1] to a colormap hex string.
    Low (< 0.3): Slate gray #718096
    Medium (0.3 - 0.7): Amber / Orange #F58518
    High (>= 0.7): Crimson / Red #E63946
    """
    imp = max(0.0, min(1.0, float(importance)))
    if imp < 0.5:
        t = imp / 0.5
        r = int(113 + t * (245 - 113))
        g = int(128 + t * (133 - 128))
        b = int(150 + t * (24 - 150))
    else:
        t = (imp - 0.5) / 0.5
        r = int(245 + t * (230 - 245))
        g = int(133 + t * (57 - 133))
        b = int(24 + t * (70 - 24))
    return f"#{r:02x}{g:02x}{b:02x}"


def node_importance_to_border_hex(base_color_hex: str, importance: float) -> str:
    """
    Compute border color hex reflecting node importance with contrast against base fill.
    """
    imp = max(0.0, min(1.0, float(importance)))
    base_hex = base_color_hex.lstrip("#")
    r = int(base_hex[0:2], 16) if len(base_hex) >= 2 else 0
    g = int(base_hex[2:4], 16) if len(base_hex) >= 4 else 0
    b = int(base_hex[4:6], 16) if len(base_hex) >= 6 else 0
    is_warm = (r > 180 and g < 160 and b < 100) or (r > 200 and g > 100 and b < 50)

    if imp >= 0.65:
        return "#ffd700" if is_warm else "#e63946"
    elif imp >= 0.30:
        return "#ffd166" if is_warm else "#f58518"
    else:
        dr = max(0, int(r * 0.6))
        dg = max(0, int(g * 0.6))
        db = max(0, int(b * 0.6))
        return f"#{dr:02x}{dg:02x}{db:02x}"


def export_pyg_explanation_to_python(
    dataset: Data,
    explanation: Explanation,
    output_dir: str = "./",
    layout: str = "spring",
    scale: int = 1000,
) -> None:
    raw_node_mask = explanation.node_mask.squeeze(-1).clone() if hasattr(explanation, "node_mask") and explanation.node_mask is not None else None
    raw_edge_mask = explanation.edge_mask.clone() if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None else None

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

    os.makedirs(output_dir, exist_ok=True)
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
            node_importance = float(raw_node_mask[node]) if raw_node_mask is not None and node < raw_node_mask.size(0) else 1.0
            border_color = node_importance_to_border_hex("#B279A2", node_importance)

            attrs = {
                "type": "Generic",
                "description": (f"Class: {y}" if hasattr(dataset, "y") else ""),
                "shape": "Circle",
                "pos": pos[node],
                "importance": round(node_importance, 4),
                "border_color": border_color,
            }

            f.write(f"G.add_node({repr(str(node))}, **{repr(attrs)})\n")


        f.write("\n")

        # -------------------------
        # Edges
        # -------------------------

        for i, edge_id in enumerate(selected_edges.tolist()):
            src = dataset.edge_index[0, edge_id].item()
            dst = dataset.edge_index[1, edge_id].item()

            if src in selected_nodes and dst in selected_nodes:
                edge_importance = float(raw_edge_mask[edge_id]) if raw_edge_mask is not None and edge_id < raw_edge_mask.size(0) else 1.0
                edge_width = round(1.5 + 4.5 * edge_importance, 2)
                edge_color = importance_to_hex(edge_importance)

                attrs = {
                    "type": "Edge",
                    "importance": round(edge_importance, 4),
                    "width": edge_width,
                    "color": edge_color,
                    "description": f"Importance: {edge_importance:.3f}",
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
    explanation hard masks, with importance scores and visual attributes preserved.
    """
    raw_node_masks = {}
    for node_type in dataset.node_types:
        if hasattr(explanation, "__getitem__") and node_type in explanation.node_types and hasattr(explanation[node_type], "node_mask"):
            nm = explanation[node_type].node_mask
            if nm is not None:
                if nm.dim() > 1:
                    nm = nm.abs().mean(dim=-1)
                raw_node_masks[node_type] = nm.clone()

    raw_edge_masks = {}
    for edge_type in explanation.edge_types:
        if hasattr(explanation[edge_type], "edge_mask") and explanation[edge_type].edge_mask is not None:
            raw_edge_masks[edge_type] = explanation[edge_type].edge_mask.clone()

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

        raw_nm = raw_node_masks.get(node_type, None)

        for idx in node_ids:
            node_id = f"{node_type}:{idx}"
            description = node_id
            if hasattr(store, "y"):
                try:
                    description += f" - Class: {int(store.y[idx])}"
                except:
                    pass

            node_importance = float(raw_nm[idx]) if raw_nm is not None and idx < raw_nm.size(0) else 1.0
            node_color = node_styles[node_type]["color"]
            border_color = node_importance_to_border_hex(node_color, node_importance)

            G.add_node(
                node_id,
                type=node_type,
                description=description,
                shape=node_styles[node_type]["shape"],
                color=node_color,
                importance=round(node_importance, 4),
                border_color=border_color,
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
        raw_em = raw_edge_masks.get(edge_type, None)

        for edge_id in edge_ids.tolist():
            src = edge_index[0, edge_id].item()
            dst = edge_index[1, edge_id].item()
            # Keep only edges between selected nodes
            if (
                src in selected_nodes[src_type]
                and
                dst in selected_nodes[dst_type]
            ):
                edge_importance = float(raw_em[edge_id]) if raw_em is not None and edge_id < raw_em.size(0) else 1.0
                edge_width = round(1.5 + 4.5 * edge_importance, 2)
                edge_color = importance_to_hex(edge_importance)

                G.add_edge(
                    f"{src_type}:{src}",
                    f"{dst_type}:{dst}",
                    type=relation,
                    importance=round(edge_importance, 4),
                    width=edge_width,
                    color=edge_color,
                    description=f"Importance: {edge_importance:.3f}",
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
