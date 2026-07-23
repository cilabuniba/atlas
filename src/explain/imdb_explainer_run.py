from .explainer_run import ExplainerRun
from src.utils import ParameterKeys
import src.models as model_pkg
import torch_geometric.datasets as data_pkg
import torch


class IMDBExplainerRun(ExplainerRun):
    def _init_general(self):
        super()._init_general()
        self.target = self.parameters.get(ParameterKeys.GENERAL).get(
            ParameterKeys.TARGET
        )

    def _init_model(self):
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_cfg = model_parameters.get(ParameterKeys.CFG, dict())
        self.model = model_pkg.__dict__[model_name](
            **model_cfg, metadata=self.dataset.metadata()
        ).to(self.device)

    def _init_loaders(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_cfg = dataset_parameters.get(ParameterKeys.CFG, dict())
        self.dataset = data_pkg.__dict__[dataset_name](**dataset_cfg)[0].to(self.device)
        self.mask = self.dataset[self.target].test_mask
        self.y = self.dataset[self.target].y

    def launch(self):
        self.model.eval()
        test_nodes = torch.where(self.mask)[0].tolist()
        iterator = self.get_bar(loader=test_nodes, desc="Explaining test nodes...")
        for node_id in iterator:
            explanation = self.explainer(
                self.dataset.x_dict,
                self.dataset.edge_index_dict,
                node=node_id,
            )
            torch.save(explanation.cpu(), f"{self.out_dir}/node_{node_id}.pt")
            self.update_bar(iterator)
        