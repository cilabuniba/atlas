from .run import Run
from src.utils import ParameterKeys
import torch_geometric.datasets as data_pkg
import src.models as model_pkg
import torch


class IMDBRun(Run):
    def _init_general(self):
        super()._init_general()
        self.target = self.parameters.get(ParameterKeys.GENERAL).get(
            ParameterKeys.TARGET
        )

    def _init_loaders(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_cfg = dataset_parameters.get(ParameterKeys.CFG, dict())
        self.dataset = data_pkg.__dict__[dataset_name](**dataset_cfg)[0].to(self.device)
        self.masks = {
            phase: getattr(self.dataset[self.target], f"{phase}_mask")
            for phase in [ParameterKeys.TRAIN, ParameterKeys.VAL, ParameterKeys.TEST]
        }
        self.y = self.dataset[self.target].y

    def _init_model(self) -> None:
        model_parameters = self.parameters.get(ParameterKeys.MODEL)
        model_name = model_parameters.get(ParameterKeys.NAME)
        model_cfg = model_parameters.get(ParameterKeys.CFG, dict())
        self.model = model_pkg.__dict__[model_name](
            **model_cfg, metadata=self.dataset.metadata()
        ).to(self.device)

    def print_stats(self, loss: float, phase: str, epoch: str = None):
        if epoch is not None:
            print(f"Epoch {epoch}/{self.num_epochs}...", end=" ")
        print(f"{phase.title()} loss: {loss:.4f}")

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        y = self.y[self.masks[ParameterKeys.TRAIN]].to(self.device)
        out = self.model(self.dataset.x_dict, self.dataset.edge_index_dict)
        out = out[self.masks[ParameterKeys.TRAIN]]
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        self.metrics.update(out.detach().cpu(), y.cpu())
        self.schedule(phase=ParameterKeys.TRAIN)
        self.print_metrics(phase=ParameterKeys.TRAIN)
        self.print_stats(
            loss=loss.detach().cpu().item(), phase=ParameterKeys.TRAIN, epoch=epoch
        )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        y = self.y[self.masks[ParameterKeys.VAL]].to(self.device)
        out = self.model(self.dataset.x_dict, self.dataset.edge_index_dict)
        out = out[self.masks[ParameterKeys.TRAIN]]
        loss = self.criterion(out, y)
        self.metrics.update(out, y)
        self.schedule(phase=ParameterKeys.VAL, epoch=epoch)
        self.trigger = self.early_stop_callback(
            cumulated_loss=loss.detach().cpu().item()
        )
        self.print_stats(
            loss=loss.detach().cpu().item(), phase=ParameterKeys.VAL, epoch=epoch
        )
        self.print_metrics(phase=ParameterKeys.VAL)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        y = self.y[self.masks[ParameterKeys.VAL]].to(self.device)
        out = self.model(self.dataset.x_dict, self.dataset.edge_index_dict)
        out = out[self.masks[ParameterKeys.TRAIN]]
        loss = self.criterion(out, y)
        self.metrics.update(out, y)
        self.print_stats(loss=loss.detach().cpu().item(), phase=ParameterKeys.TEST)
        self.print_metrics(phase=ParameterKeys.TEST)
