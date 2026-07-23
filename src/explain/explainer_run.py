from src.utils import ParameterKeys
import src.models as model_pkg
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
import src.explain.algorithms as exp_algo_pkg
from torch_geometric.explain import Explainer


class ExplainerRun:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        print("Init general parameters")
        self._init_general()
        print("Init test datalaoder")
        self._init_loaders()
        print("Init model")
        self._init_model()
        print("Init explainer")
        self._init_explainer()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL)
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.device = general_parameters.get(ParameterKeys.DEVICE, "cpu")
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.state_dict_path = general_parameters.get(
            ParameterKeys.STATE_DICT, "./model.pt"
        )

    def _init_loaders(self) -> None:
        raise NotImplementedError()

    def _init_model(self):
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_config = model_parameters.get(ParameterKeys.CFG, {})
        self.model = model_pkg.__dict__[model_name](**model_config)
        self.model = self.model.to(self.device)

    def _init_explainer(self) -> None:
        explainer_parameters = self.parameters.get(ParameterKeys.EXPLAINER)
        algorithm_parameters = explainer_parameters.pop(ParameterKeys.ALGORITHM)
        algorithm_name = algorithm_parameters.get(ParameterKeys.NAME)
        algorithm_config = algorithm_parameters.get(ParameterKeys.CFG, dict())
        algorithm = exp_algo_pkg.__dict__[algorithm_name](**algorithm_config)
        self.explainer = Explainer(
            model=self.model, algorithm=algorithm, **explainer_parameters
        )

    def load_state_dict(self) -> None:
        state_dict = torch.load(self.state_dict_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

    def get_bar(self, loader: DataLoader, desc: str = "", **kwargs):
        return (
            loader
            if not self.pbar
            else tqdm(
                loader,
                total=len(loader),
                desc=desc,
                **kwargs,
            )
        )

    def update_bar(self, bar, **kwargs):
        if not self.pbar:
            return
        bar.set_postfix(**kwargs)

    def launch(self) -> None:
        raise NotImplementedError()
