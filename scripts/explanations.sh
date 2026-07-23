# !/bin/bash

uv run script_exe.py explain --parameters configs/explain/imdb.yaml --cls IMDBExplainerRun
uv run script_exe.py explain --parameters configs/explain/movielens.yaml --cls MovielensExplainerRun
uv run script_exe.py explain --parameters configs/explain/planetoid.yaml --cls PlanetoidExplainerRun
uv run script_exe.py explain --parameters configs/explain/ppi.yaml --cls PPIExplainerRun
uv run script_exe.py explain --parameters configs/explain/wikics.yaml --cls WikiCSExplainerRun
