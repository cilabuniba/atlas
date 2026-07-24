from .explainer_run import ExplainerRun
from src.utils import ParameterKeys
import torch_geometric.datasets as data_pkg
import src.models as model_pkg
import torch
from torch_geometric.explain import Explanation
import os


class PPIExplainerRun(ExplainerRun):
    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_cfg = dataset_parameters.get(ParameterKeys.CFG, dict())
        self.dataset = data_pkg.__dict__[dataset_name](**dataset_cfg)

    def postprocess_explanation(self, explanation: Explanation) -> Explanation:
        out_explanation = explanation.clone()
        del out_explanation.x
        del out_explanation.edge_index
        return out_explanation

    def launch(self):
        self.model.eval()
        self.load_state_dict()
        iterator = self.get_bar(loader=self.dataset, desc="Explaining test graphs")
        for ix, graph in enumerate(iterator):
            graph = graph.to(self.device)
            out = self.model(graph.x, graph.edge_index).sigmoid() > 0.5
            for node_id in range(graph.num_nodes):
                print(f"Doing node {node_id}")
                for class_idx in range(out.size(1)):
                    explanation_path = (
                        f"{self.out_dir}/graph_{ix}_node_{node_id}_class_{class_idx}.pt"
                    )
                    if not out[node_id, class_idx].cpu().item() or os.path.exists(
                        explanation_path
                    ):
                        continue
                    print(f"Doing class {class_idx}")
                    explanation = self.explainer(
                        graph.x,
                        graph.edge_index,
                        node=node_id,
                        class_idx=class_idx,
                    )
                    torch.save(
                        self.postprocess_explanation(explanation.cpu()),
                        explanation_path,
                    )
            self.update_bar(iterator)
