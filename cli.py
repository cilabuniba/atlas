import click
from utils.io import load_ruamel

@click.group()
def main():
    pass


@main.command("dataset_code")
@click.option("--parameters", help="Path to the parameters file", required=True)
def dataset_code(parameters):
    from dataset_download import generate_dataset_code
    return generate_dataset_code(load_ruamel(parameters))
