# !/bin/bash

uv run script_exe.py training --parameters configs/training/imdb.yaml --cls IMDBRun
uv run script_exe.py training --parameters configs/training/movielens.yaml --cls MovielensRun
uv run script_exe.py training --parameters configs/training/planetoid.yaml --cls PlanetoidRun
uv run script_exe.py training --parameters configs/training/ppi.yaml --cls PPIRun
uv run script_exe.py training --parameters configs/training/wikics.yaml --cls WikiCSRun
