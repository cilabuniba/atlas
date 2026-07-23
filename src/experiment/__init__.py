from .imdb_run import IMDBRun
from .movielens_run import MovielensRun
from .planetoid_run import PlanetoidRun
from .ppi_run import PPIRun
from .wikics_run import WikiCSRun


def exe_experiment(parameters: dict, cls: str):
    run_cls = globals()[cls]
    run_cls(parameters).launch()