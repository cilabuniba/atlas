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
def dataset_code(parameters, hetero):
    from src.dataset_download import generate_code
    return generate_code(load_ruamel(parameters), hetero)


@main.command("training")
@click.option("--parameters", help="Path to the parameters file", required=True)
@click.option(
    "--cls", help="Class to be used for running the experiment", required=True
)
def training(parameters, cls):
    from src.experiment import exe_experiment
    exe_experiment(load_ruamel(parameters), cls)
