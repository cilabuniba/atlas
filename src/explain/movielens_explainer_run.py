from .explainer_run import ExplainerRun
from src.utils import ParameterKeys
import torch_geometric.datasets as data_pkg
from torch_geometric.transforms import RandomLinkSplit
import src.models as model_pkg
import torch
from torch_geometric.explain import HeteroExplanation
import os


class MovielensExplainerRun(ExplainerRun):
    def _init_general(self):
        super()._init_general()
        self.target = self.parameters.get(ParameterKeys.GENERAL).get(
            ParameterKeys.TARGET
        )

    def _init_model(self) -> None:
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_cfg = model_parameters.get(ParameterKeys.CFG, dict())
        self.model = model_pkg.__dict__[model_name](
            **model_cfg, metadata=self.dataset.metadata()
        ).to(self.device)

    def _init_loaders(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG)
        split_config = dataset_config.pop(ParameterKeys.SPLIT)
        split_config["edge_types"] = tuple(split_config["edge_types"])
        split_config["rev_edge_types"] = tuple(split_config["rev_edge_types"])
        dataset = data_pkg.__dict__[dataset_name](**dataset_config)[0].to(self.device)
        del dataset[*self.target].edge_label_index
        del dataset[*self.target].edge_label
        transform = RandomLinkSplit(**split_config)
        _, _, self.dataset = transform(dataset)

    def postprocess_explanation(
        self, explanation: HeteroExplanation
    ) -> HeteroExplanation:
        out_explanation = explanation.clone()
        for node_type in out_explanation.node_types:
            del out_explanation[node_type].x
        for edge_type in out_explanation.edge_types:
            del out_explanation[edge_type].edge_index
        return out_explanation

    def launch(self) -> None:
        self.model.eval()
        self.load_state_dict()
        test_edges = self.dataset[*self.target].edge_label_index.T.tolist()
        iterator = self.get_bar(
            loader=list(enumerate(test_edges)), desc="Explaining test edeges..."
        )
        for edge_idx, (src, dst) in iterator:
            explanation_path = f"{self.out_dir}/edge_{src}_{dst}.pt"
            if os.path.exists(explanation_path):
                continue
            explanation = self.explainer(
                self.dataset.x_dict,
                self.dataset.edge_index_dict,
                edge_label_index=self.dataset[*self.target].edge_label_index,
                edge_label_type=(tuple(self.target), edge_idx),
            )
            torch.save(
                self.postprocess_explanation(explanation.cpu()), explanation_path
            )
            self.update_bar(iterator)
