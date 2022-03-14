# %% load libraries
import json
import dask
import dask.dataframe as dd
import os
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


# %% load data in
# data = json.load(open("./data/spotify_million_playlist_dataset/data/mpd.slice.0-999.json"))
# df = pd.DataFrame(data["playlists"])
dd.read_json("./data/spotify_million_playlist_dataset/data/mpd.slice.0-999.json", lines=True)
print("loaded json in dataframe")

# %% load data in
print("hello World")