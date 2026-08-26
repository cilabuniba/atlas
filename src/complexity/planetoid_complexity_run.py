from .complexity_run import ComplexityRun


class PlanetoidComplexityRun(ComplexityRun):
    def _init_data(self) -> None:
        super()._init_data()
        self.mode_type = "Homogeneous"
        if not self.data_root:
            if self.num_nodes is not None:
                self.data_root = f"dataset_code/metrics/planetoid/{self.num_nodes}/graph_export.py"
            else:
                self.data_root = "dataset_code/metrics/planetoid"
