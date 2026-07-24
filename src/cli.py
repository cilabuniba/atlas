import click
from src.utils.io import load_ruamel


@click.group()
def main():
    pass


@main.command("dataset_code")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option(
    "--hetero",
    is_flag=True,
    help="Optional flag to impose heterogeneous graph filtering",
)
@click.option(
    "--explanation",
    is_flag=True,
    help="Optional flag to impose explanation export",
)
def dataset_code(parameters, hetero, explanation):
    from src.code_gen import generate_code

    return generate_code(load_ruamel(parameters), hetero, explanation)


@main.command("training")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option(
    "--cls", help="Class to be used for running the experiment", required=True
)
def training(parameters, cls):
    from src.experiment import exe_experiment

    exe_experiment(load_ruamel(parameters), cls)


@main.command("explain")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option(
    "--cls", help="Class to be used for running the experiment", required=True
)
def explain(parameters, cls):
    from src.explain import exe_explain

    exe_explain(load_ruamel(parameters), cls)
