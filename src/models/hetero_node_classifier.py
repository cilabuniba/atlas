import torch
from torch_geometric.nn import to_hetero
import torch_geometric.nn.models as backbone_pkg


class HeteroNodeClassifier(torch.nn.Module):
    def __init__(
        self,
        gnn_name: str,
        gnn_config: dict,
        target_node_type: str,
        hidden_dim: int,
        num_classes: int,
        metadata,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        self.gnn = backbone_pkg.__dict__[gnn_name](**gnn_config)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr=aggr)
        self.target_node_type = target_node_type
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x_dict, edge_index_dict, **kwargs) -> torch.Tensor:
        node = kwargs.pop("node", None)
        embs = self.gnn(x_dict, edge_index_dict, **kwargs)
        embs = embs[self.target_node_type]
        embs = embs[node] if node is not None else embs
        return self.linear(embs)