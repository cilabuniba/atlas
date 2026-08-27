from .complexity_run import ComplexityRun


class MovielensComplexityRun(ComplexityRun):
    def _init_data(self) -> None:
        super()._init_data()
        self.mode_type = "Heterogeneous"
        if not self.data_root:
            if self.num_nodes is not None:
                self.data_root = f"dataset_code/metrics/movielens100k/{self.num_nodes}/graph_export.py"
            else:
                self.data_root = "dataset_code/metrics/movielens100k"
