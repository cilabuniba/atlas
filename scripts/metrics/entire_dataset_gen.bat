uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\planetoid\100.yaml
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\planetoid\500.yaml
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\planetoid\1000.yaml
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\planetoid\2500.yaml
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\planetoid\5000.yaml
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\planetoid\10000.yaml

uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\movielens\100.yaml --hetero
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\movielens\500.yaml --hetero
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\movielens\1000.yaml --hetero
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\movielens\2500.yaml --hetero
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\movielens\5000.yaml --hetero
uv run script_exe.py dataset_code --parameters configs\metrics\dataset_download\movielens\10000.yaml --hetero