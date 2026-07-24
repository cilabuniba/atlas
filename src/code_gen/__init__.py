from .entire_dataset_code_gen import generate_dataset_code, generate_hetero_dataset_code
from .explanation_code_gen import generate_explanation_code


def generate_code(parameters: dict, hetero: bool = False, explanation: bool = False):
    if explanation:
        if hetero:
            raise NotImplementedError()
        else:
            generate_explanation_code(parameters)
    elif hetero:
        generate_hetero_dataset_code(parameters)
    else:
        generate_dataset_code(parameters)
