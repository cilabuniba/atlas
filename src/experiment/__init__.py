from .imdb_run import IMDBRun
from .movielens_run import MovielensRun
from .planetoid_run import PlanetoidRun


def exe_experiment(parameters: dict, cls: str):
    run_cls = globals()[cls]
    run_cls(parameters).launch()