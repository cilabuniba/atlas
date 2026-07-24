from .run import Run
from src.utils import ParameterKeys
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.datasets as data_pkg
import src.models as model_pkg
import torch


class MovielensRun(Run):
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
            **model_cfg, metadata=self.train_data.metadata()
        ).to(self.device)

    def _init_loaders(self) -> None:
        dataset_parameters = self.parameters.get(ParameterKeys.DATA)
        dataset_name = dataset_parameters.get(ParameterKeys.NAME)
        dataset_config = dataset_parameters.get(ParameterKeys.CFG)
        split_config = dataset_config.pop(ParameterKeys.SPLIT)
        split_config["edge_types"] = tuple(split_config["edge_types"])
        split_config["rev_edge_types"] = tuple(split_config["rev_edge_types"])
        dataset = data_pkg.__dict__[dataset_name](**dataset_config)[0] #.to(self.device)
        del dataset[*self.target].edge_label_index
        del dataset[*self.target].edge_label
        transform = RandomLinkSplit(**split_config)
        self.train_data, self.val_data, self.test_data = transform(dataset)
        torch.save(self.test_data, f"{self.out_dir}/test_dataset.pt")

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        self.train_data = self.train_data.to(self.device)
        out = self.model(
            self.train_data.x_dict,
            self.train_data.edge_index_dict,
            self.train_data[*self.target].edge_label_index
        )
        labels = (self.train_data[*self.target].edge_label.long() - 1).to(self.device)
        loss = self.criterion(out, labels)
        loss.backward()
        self.optimizer.step()
        self.metrics.update(out.detach().cpu(), labels.cpu())
        self.schedule(phase=ParameterKeys.TRAIN)
        self.print_metrics(phase=ParameterKeys.TRAIN)
        self.print_stats(
            loss=loss.detach().cpu().item(), phase=ParameterKeys.TRAIN, epoch=epoch
        )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        self.val_data = self.val_data.to(self.device)
        out = self.model(
            self.val_data.x_dict,
            self.val_data.edge_index_dict,
            self.val_data[*self.target].edge_label_index,
        )
        labels = self.val_data[*self.target].edge_label.long() - 1
        loss = self.criterion(out, labels)
        self.metrics.update(out.detach().cpu(), labels.cpu())
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
        self.test_data = self.test_data.to(self.device)
        out = self.model(
            self.test_data.x_dict,
            self.test_data.edge_index_dict,
            self.test_data[*self.target].edge_label_index,
        )
        labels = self.test_data[*self.target].edge_label.long() - 1
        loss = self.criterion(out, labels)
        self.metrics.update(out.detach().cpu(), labels.cpu())
        self.print_stats(loss=loss.detach().cpu().item(), phase=ParameterKeys.TEST)
        self.print_metrics(phase=ParameterKeys.TEST)
