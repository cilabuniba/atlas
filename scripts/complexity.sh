# !/bin/bash

# Planetoid
uv run script_exe.py complexity --parameters configs/complexity/planetoid/100.yaml --cls PlanetoidComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/planetoid/500.yaml --cls PlanetoidComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/planetoid/1000.yaml --cls PlanetoidComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/planetoid/2500.yaml --cls PlanetoidComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/planetoid/5000.yaml --cls PlanetoidComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/planetoid/10000.yaml --cls PlanetoidComplexityRun

# Movielens100k
uv run script_exe.py complexity --parameters configs/complexity/movielens100k/100.yaml --cls MovielensComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/movielens100k/500.yaml --cls MovielensComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/movielens100k/1000.yaml --cls MovielensComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/movielens100k/2500.yaml --cls MovielensComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/movielens100k/5000.yaml --cls MovielensComplexityRun
uv run script_exe.py complexity --parameters configs/complexity/movielens100k/10000.yaml --cls MovielensComplexityRun
