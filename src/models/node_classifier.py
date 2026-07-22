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
        embs = self.gnn(x, edge_index, **kwargs)
        return self.linear(embs)
