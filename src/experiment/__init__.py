from .imdb_run import IMDBRun


def exe_experiment(parameters: dict, cls: str):
    run_cls = globals()[cls]
    run_cls(parameters).launch()