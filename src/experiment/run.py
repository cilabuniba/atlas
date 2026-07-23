from src.utils import ParameterKeys
import src.models as models
import src.losses as losses
import os
import torch.optim as optimizers
import src.early_stop as early_stops
import src.scheduler as schedulers
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
import src.metrics as metric_pkg
import torch


class Run:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        print("Init general parameters")
        self._init_general()
        print("Init dataloaders")
        self._init_loaders()
        print("Init model")
        self._init_model()
        print("Init optimizer")
        self._init_optimizer()
        print("Init criterion")
        self._init_criterion()
        print("Init metrics")
        self._init_metrics()
        print("Init eearly stop")
        self._init_early_stop()
        print("Init scheduler")
        self._init_scheduler()

    def _init_general(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, {})
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.device = general_parameters.get(ParameterKeys.DEVICE, "cpu")
        self.device = self.device if torch.cuda.is_available() else "cpu"
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.num_epochs = general_parameters.get(ParameterKeys.NUM_EPOCHS, 1)

    def _init_loaders(self):
        raise NotImplementedError()

    def _init_model(self):
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_config = model_parameters.get(ParameterKeys.CFG, {})
        self.model = models.__dict__[model_name](**model_config)
        self.model = self.model.to(self.device)

    def _init_optimizer(self):
        optimizer_parameters = self.parameters.get(ParameterKeys.OPTIMIZER)
        optimizer_name = optimizer_parameters.get(ParameterKeys.NAME)
        optimizer_cfg = optimizer_parameters.get(ParameterKeys.CFG, {})
        self.optimizer = optimizers.__dict__[optimizer_name](
            params=self.model.parameters(), **optimizer_cfg
        )

    def _init_metrics(self):
        metrics_parameters = self.parameters.get(ParameterKeys.METRICS, dict())
        self.metrics = MetricCollection(
            {
                k: metric_pkg.__dict__[v.get(ParameterKeys.NAME)](
                    **v.get(ParameterKeys.CFG, dict())
                )
                for k, v in metrics_parameters.items()
            }
        )

    def _init_criterion(self):
        criterion_parameters = self.parameters.get(ParameterKeys.CRITERION, {})
        criterion_name = criterion_parameters.get(ParameterKeys.NAME)
        criterion_cfg = criterion_parameters.get(ParameterKeys.CFG, {})
        if "weight" in criterion_cfg:
            criterion_cfg["weight"] = torch.as_tensor(criterion_cfg["weight"])
        self.criterion = losses.__dict__[criterion_name](**criterion_cfg)

    def _init_early_stop(self):
        early_stop_parameters = self.parameters.get(ParameterKeys.EARLY_STOP, None)
        if early_stop_parameters is None:
            self.early_stop = None
            return
        early_stop_name = early_stop_parameters.get(ParameterKeys.NAME)
        early_stop_cfg = early_stop_parameters.get(ParameterKeys.CFG, {})
        self.early_stop = early_stops.__dict__[early_stop_name](**early_stop_cfg)

    def _init_scheduler(self):
        scheduler_parameters = self.parameters.get(ParameterKeys.SCHEDULER, None)
        if scheduler_parameters is None:
            self.scheduler = None
            return
        scheduler_name = scheduler_parameters.get(ParameterKeys.NAME)
        scheduler_cfg = scheduler_parameters.get(ParameterKeys.CFG, {})
        if scheduler_name == ParameterKeys.HF_COSINE_SCHEDULER:
            scheduler_cfg = self._init_cosine_scheduler_params(params=scheduler_cfg)
            self.cosine_sched = True
        else:
            self.cosine_sched = False
        self.scheduler = schedulers.__dict__[scheduler_name](
            optimizer=self.optimizer, **scheduler_cfg
        )

    def _init_cosine_scheduler_params(self, params: dict) -> schedulers.LambdaLR:
        total_steps = (
            self.num_epochs * len(self.train_loader)
            if hasattr(self, "train_loader")
            else self.num_epochs
        )
        params.update({"num_training_steps": total_steps})
        return params

    def schedule(self, phase, **kwargs):
        if self.scheduler is None:
            return
        if phase == ParameterKeys.VAL and self.cosine_sched:
            return
        if phase == ParameterKeys.TRAIN and not self.cosine_sched:
            return
        self.scheduler.step(**kwargs)

    def early_stop_callback(self, cumulated_loss: float) -> bool:
        if self.early_stop is None:
            return False
        self.early_stop(cumulated_loss, self.model)
        return self.early_stop.early_stop

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

    def update_bar(self, bar, loss, **kwargs):
        if not self.pbar:
            return
        bar.set_postfix({ParameterKeys.CRITERION: loss, **kwargs})

    def print_stats(self, loss: float, phase: str, epoch: str = None):
        if epoch is not None:
            print(f"Epoch {epoch}/{self.num_epochs}...", end=" ")
        print(f"{phase.title()} loss: {loss:.4f}")

    def train_epoch(self, epoch: int):
        raise NotImplementedError()

    def validate_epoch(self, epoch: int):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def launch(self) -> None:
        print("Start experiment!")
        self.model = self.model.to(self.device)
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            self.validate_epoch(epoch=epoch)
            if self.trigger:
                print(f"Early stopping at epoch {epoch}/{self.num_epochs}")
                break
        self.load_state_dict()
        return self.test()
    
    def load_state_dict(self):
        state_dict_path = f"{self.out_dir}/model.pt"
        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def print_metrics(self, phase: str):
        metrics = self.metrics.compute()
        for name, val in metrics.items():
            print(f"{phase} {name}: {val.item():.4f}")
        self.metrics.reset()
