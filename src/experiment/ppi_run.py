from .run import Run
from src.utils import ParameterKeys
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch_geometric.datasets as data_pkg
import torch


class PPIRun(Run):
    def _init_dataset(self, dataset_parameters: dict) -> InMemoryDataset:
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_cfg = dataset_parameters.get(ParameterKeys.CFG, dict())
        return data_pkg.__dict__[dataset_name](**dataset_cfg)

    def _init_loaders(self):
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        train_dataset = self._init_dataset(dataset_parameters.get(ParameterKeys.TRAIN))
        val_dataset = self._init_dataset(dataset_parameters.get(ParameterKeys.VAL))
        test_dataset = self._init_dataset(dataset_parameters.get(ParameterKeys.TEST))
        loader_parameters = self.parameters.get(ParameterKeys.LOADER, dict())

        self.train_loader = DataLoader(dataset=train_dataset, **loader_parameters)
        self.val_loader = DataLoader(dataset=val_dataset, **loader_parameters)
        self.test_loader = DataLoader(dataset=test_dataset, **loader_parameters)

    def train_epoch(self, epoch):
        self.model.train()
        loader = self.get_bar(
            loader=self.train_loader,
            desc=f"{ParameterKeys.TRAIN} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.metrics.update(out.detach().cpu(), batch.y.cpu())
            self.update_bar(bar=loader, loss=batch_loss)
            self.schedule(phase=ParameterKeys.TRAIN)
        cumulated_loss = cumulated_loss / len(self.train_loader)
        self.print_metrics(phase=ParameterKeys.TRAIN)
        self.print_stats(loss=cumulated_loss, phase=ParameterKeys.TRAIN, epoch=epoch)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        loader = self.get_bar(
            loader=self.val_loader,
            desc=f"{ParameterKeys.VAL} at epoch {epoch}/{self.num_epochs}",
        )
        cumulated_loss = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out, batch.y)
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.metrics.update(out.cpu(), batch.y.cpu())
            self.update_bar(bar=loader, loss=batch_loss)
        self.schedule(phase=ParameterKeys.VAL, epoch=epoch)
        cumulated_loss = cumulated_loss / len(self.val_loader)
        self.trigger = self.early_stop_callback(cumulated_loss=cumulated_loss)
        self.print_metrics(phase=ParameterKeys.VAL)
        self.print_stats(loss=cumulated_loss, phase=ParameterKeys.VAL, epoch=epoch)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        loader = self.get_bar(
            loader=self.test_loader,
            desc=f"{ParameterKeys.TEST}",
        )
        cumulated_loss = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out, batch.y)
            batch_loss = loss.detach().cpu().item()
            cumulated_loss += batch_loss
            self.metrics.update(out.cpu(), batch.y.cpu())
            self.update_bar(bar=loader, loss=batch_loss)
        cumulated_loss = cumulated_loss / len(self.test_loader)
        self.print_metrics(phase=ParameterKeys.VAL)
        self.print_stats(loss=cumulated_loss, phase=ParameterKeys.VAL)
