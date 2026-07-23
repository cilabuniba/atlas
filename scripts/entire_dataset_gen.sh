# !/bin/bash

uv run script_exe.py dataset_code --parameters configs/dataset_download/imdb.yaml --hetero
uv run script_exe.py dataset_code --parameters configs/dataset_download/movielens.yaml --hetero
uv run script_exe.py dataset_code --parameters configs/dataset_download/opd_dataset.yaml --hetero
uv run script_exe.py dataset_code --parameters configs/dataset_download/planetoid.yaml 
uv run script_exe.py dataset_code --parameters configs/dataset_download/ppi.yaml
uv run script_exe.py dataset_code --parameters configs/dataset_download/wikics.yaml  
