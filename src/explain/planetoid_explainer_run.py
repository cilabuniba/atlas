from .explainer_run import ExplainerRun
import torch_geometric.datasets as data_pkg
import src.models as model_pkg
from src.utils import ParameterKeys
import torch


class PlanetoidExplainerRun(ExplainerRun):
    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG, dict())
        self.dataset = data_pkg.__dict__[dataset_name](**dataset_config)[0].to(
            self.device
        )
        self.mask = self.dataset.test_mask
        self.y = self.dataset.y

    def launch(self):
        self.model.eval()
        self.load_state_dict()
        test_nodes = torch.where(self.mask)[0].tolist()
        iterator = self.get_bar(loader=test_nodes, desc="Explaining test nodes...")
        for node_id in iterator:
            explanation = self.explainer(
                self.dataset.x,
                self.dataset.edge_index,
                node=node_id,
            )
            torch.save(explanation.cpu(), f"{self.out_dir}/node_{node_id}.pt")
            self.update_bar(iterator)