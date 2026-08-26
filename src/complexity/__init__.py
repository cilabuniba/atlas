from .complexity_run import ComplexityRun
from .planetoid_complexity_run import PlanetoidComplexityRun
from .movielens_complexity_run import MovielensComplexityRun
from .imdb_complexity_run import IMDBComplexityRun
from .ppi_complexity_run import PPIComplexityRun
from .wikics_complexity_run import WikiCSComplexityRun


def exe_complexity(parameters: dict, cls: str):
    run_cls = globals()[cls]
    run_cls(parameters).launch()
