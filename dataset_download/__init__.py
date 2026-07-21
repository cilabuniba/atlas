from .entire_dataset_code_gen import generate_dataset_code, generate_hetero_dataset_code



def generate_code(parameters: dict, hetero: bool = False):
    if hetero:
        generate_hetero_dataset_code(parameters)
    else:
        generate_dataset_code(parameters)
