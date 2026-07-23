from .imdb_explainer_run import IMDBExplainerRun
from .movielens_explainer_run import MovielensExplainerRun
from .planetoid_explainer_run import PlanetoidExplainerRun
from .ppi_explainer_run import PPIExplainerRun
from .wikics_explainer_run import WikiCSExplainerRun


def exe_explain(parameters: dict, cls: str):
    run_cls = globals()[cls]
    run_cls(parameters).launch()