import pandas as pd
import os
import numpy as np
import json
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sys import getsizeof
from itertools import chain


data = json.load(open('../../ML_AI/ds_capstone/data/mpd.slice.0-999.json'))

db = []
for row in data["playlists"]:
    #p_id = row["pid"]
    tracks = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
    db.append(tracks)

# get unique songs
song_list = set(chain(*db))

mx = []

for row in tqdm(db):
    temp = []
    for s in song_list:
        if s in row:
            temp.append(1)
        else:
            temp.append(0)

    mx.append(temp)


sparseMatrix = csr_matrix(mx)

# check sizes

print(getsizeof(mx))

print(getsizeof(sparseMatrix))