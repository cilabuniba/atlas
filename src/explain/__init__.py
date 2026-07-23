from .imdb_explainer_run import IMDBExplainerRun


def exe_explain(parameters: dict, cls: str):
    run_cls = globals()[cls]
    run_cls(parameters).launch()