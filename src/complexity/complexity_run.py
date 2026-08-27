import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from src.utils import ParameterKeys

# Ensure Qt offscreen platform if running in headless environment
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from views.graph_scene import GraphScene
from widgets.metrics_panel import MetricsPanel
from utils.code_importer import GraphImporter

# Global reference to prevent garbage collection of QApplication
_QAPP_INSTANCE = None


def get_or_create_qapp() -> QApplication:
    global _QAPP_INSTANCE
    app = QApplication.instance()
    if app is None:
        _QAPP_INSTANCE = QApplication(["atlas"])
        app = _QAPP_INSTANCE
    return app


class ComplexityRun:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.init()

    def init(self) -> None:
        print("Init general parameters")
        self._init_general()
        print("Init data parameters")
        self._init_data()
        print("Init graph scene")
        self._init_scene()

    def _init_general(self) -> None:
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, {})
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "data/metrics/results")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = general_parameters.get(ParameterKeys.PBAR, False)
        self.compute_metrics = general_parameters.get("compute_metrics", False)

    def _init_data(self) -> None:
        data_parameters = self.parameters.get(ParameterKeys.DATA, {})
        self.data_root = data_parameters.get("root", data_parameters.get("path", ""))
        self.num_nodes = data_parameters.get("num_nodes", None)
        self.node_counts = data_parameters.get("node_counts", None)
        self.mode_type = data_parameters.get("mode_type", "Homogeneous")

    def _init_scene(self) -> None:
        self.app = get_or_create_qapp()
        self.scene = GraphScene(mode_type=self.mode_type)
        if self.compute_metrics:
            self.metrics_panel = MetricsPanel(self.scene)
            self.scene.set_metrics_callback(self.metrics_panel.update_metrics)
        else:
            self.metrics_panel = None
            self.scene.set_metrics_callback(None)
            self.scene.update_metrics = lambda: None


    def benchmark_code(
        self,
        code: str,
        output_metrics_file: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        temp_file = None
        target_path = output_metrics_file
        if target_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            target_path = temp_file.name
            temp_file.close()
        else:
            target_path = str(target_path)
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        try:
            GraphImporter.import_from_code(code, self.scene, metrics_filename=target_path)
            with open(target_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        finally:
            if temp_file is not None and os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass
            self.scene.clear_all()

        return metrics_data

    def benchmark_file(
        self,
        graph_file: str | Path,
        output_metrics_file: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        graph_path = Path(graph_file)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        with open(graph_path, "r", encoding="utf-8") as f:
            code = f.read()

        metrics = self.benchmark_code(code, output_metrics_file)
        metrics["file"] = str(graph_path)
        metrics["mode_type"] = self.mode_type
        return metrics

    def launch(self) -> List[Dict[str, Any]]:
        target_root = Path(self.data_root)
        if not target_root.exists():
            export_params = self.parameters.get("export", {})
            if "output_dir" in export_params:
                target_root = Path(export_params["output_dir"])

        if not target_root.exists():
            raise FileNotFoundError(f"Target data path not found: {self.data_root}")

        results = []

        if target_root.is_file() and target_root.suffix == ".py":
            node_label = str(self.num_nodes) if self.num_nodes is not None else (
                target_root.parent.name if target_root.parent.name.isdigit() else "metrics"
            )
            out_file = Path(self.out_dir) / f"{node_label}.json"
            print(f"--> Benchmarking single file {target_root} (nodes={node_label})...")
            res = self.benchmark_file(target_root, output_metrics_file=out_file)
            res["node_count_target"] = int(node_label) if node_label.isdigit() else node_label
            results.append(res)
            print(
                f"    Result: nodes={res['number_of_nodes']}, edges={res['number_of_edges']}, "
                f"time_parsing={res['time_of_parsing']:.4f}s, "
                f"time_rendering={res['time_of_rendering']:.4f}s, "
                f"time_metrics={res['time_for_metric_computation']:.4f}s, "
                f"peak_mem={res['memory_peak_mb']:.2f}MB"
            )
        elif (target_root / "graph_export.py").exists():
            graph_file = target_root / "graph_export.py"
            node_label = str(self.num_nodes) if self.num_nodes is not None else (
                target_root.name if target_root.name.isdigit() else "metrics"
            )
            out_file = Path(self.out_dir) / f"{node_label}.json"
            print(f"--> Benchmarking {graph_file} (nodes={node_label})...")
            res = self.benchmark_file(graph_file, output_metrics_file=out_file)
            res["node_count_target"] = int(node_label) if node_label.isdigit() else node_label
            results.append(res)
            print(
                f"    Result: nodes={res['number_of_nodes']}, edges={res['number_of_edges']}, "
                f"time_parsing={res['time_of_parsing']:.4f}s, "
                f"time_rendering={res['time_of_rendering']:.4f}s, "
                f"time_metrics={res['time_for_metric_computation']:.4f}s, "
                f"peak_mem={res['memory_peak_mb']:.2f}MB"
            )
        else:
            subdirs = [p for p in target_root.iterdir() if p.is_dir()]
            subdirs.sort(key=lambda p: int(p.name) if p.name.isdigit() else p.name)

            if self.node_counts:
                allowed = set(str(n) for n in self.node_counts)
                subdirs = [p for p in subdirs if p.name in allowed]

            iterator = tqdm(subdirs, desc="Benchmarking graphs") if self.pbar else subdirs

            for node_dir in iterator:
                graph_file = node_dir / "graph_export.py"
                if not graph_file.exists():
                    continue

                node_label = node_dir.name
                metrics_out = Path(self.out_dir) / f"{node_label}.json"

                print(f"--> Benchmarking nodes={node_label} from {graph_file}...")
                metrics = self.benchmark_file(
                    graph_file=graph_file,
                    output_metrics_file=metrics_out,
                )
                metrics["node_count_target"] = int(node_label) if node_label.isdigit() else node_label
                results.append(metrics)
                print(
                    f"    Result: nodes={metrics['number_of_nodes']}, edges={metrics['number_of_edges']}, "
                    f"time_parsing={metrics['time_of_parsing']:.4f}s, "
                    f"time_rendering={metrics['time_of_rendering']:.4f}s, "
                    f"time_metrics={metrics['time_for_metric_computation']:.4f}s, "
                    f"peak_mem={metrics['memory_peak_mb']:.2f}MB"
                )

            summary_file = Path(self.out_dir) / "summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"\nAll benchmark results saved to {self.out_dir}")

        return results
