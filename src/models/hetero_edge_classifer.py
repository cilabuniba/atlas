import torch
from torch_geometric.nn import to_hetero
import torch_geometric.nn.models as backbone_pkg


class HeteroEdgeClassifier(torch.nn.Module):
    def __init__(
        self,
        gnn_name: str,
        gnn_config: dict,
        src_nodet: str,
        dst_nodet: str,
        hidden_dim: int,
        num_classes: int,
        metadata,
        aggr: str,
    ):
        super().__init__()
        self.gnn = backbone_pkg.__dict__[gnn_name](**gnn_config)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr=aggr)
        self.src_nodet = src_nodet
        self.dst_nodet = dst_nodet
        self.linear = torch.nn.Linear(in_features=hidden_dim*2, out_features=num_classes)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index, **kwargs) -> torch.Tensor:
        embs = self.gnn(x_dict, edge_index_dict, **kwargs)
        src = embs[self.src_nodet][edge_label_index[0]]
        dst = embs[self.dst_nodet][edge_label_index[1]]
        return self.linear(torch.cat([src, dst], dim=-1))
    
