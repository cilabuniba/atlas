from .complexity_run import ComplexityRun


class PPIComplexityRun(ComplexityRun):
    def _init_data(self) -> None:
        super()._init_data()
        self.mode_type = "Homogeneous"
        if not self.data_root:
            self.data_root = "dataset_code/metrics/ppi"
