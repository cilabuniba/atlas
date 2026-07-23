import torch
import torch_geometric.nn.models as backbone_pkg


class NodeClassifier(torch.nn.Module):
    def __init__(
        self,
        gnn_name: str,
        gnn_config: dict,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.gnn = backbone_pkg.__dict__[gnn_name](**gnn_config)
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x, edge_index, **kwargs) -> torch.Tensor:
        node_id = kwargs.pop("node", None)
        class_idx = kwargs.pop("class_idx", None)
        embs = self.gnn(x, edge_index, **kwargs)
        embs = embs[node_id] if node_id is not None else embs
        logits = self.linear(embs)
        if class_idx is not None:
            logits = logits[:, class_idx] if logits.ndim > 1 else logits[class_idx]
        return logits
