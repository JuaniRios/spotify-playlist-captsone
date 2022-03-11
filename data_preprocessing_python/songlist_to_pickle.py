import json
import os
from itertools import chain
import time
import pickle


def get_songlist(n, filename):
    counter = 0
    start = time.time()
    db = []
    for file in os.listdir("../../ML_AI/ds_capstone/data"):
        if counter == n:
            break
        data = json.load(open('../../ML_AI/ds_capstone/data/'+file))
        for row in data["playlists"]:
            tracks = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
            db.append(tracks)
        counter += 1

    song_list = list(set(chain(*db)))
    song_map = {}
    for i, song in enumerate(song_list):
        song_map[song] = i

    with open(filename, 'wb') as f:
        pickle.dump(song_map, f, pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print("Total runtime of creating pickle: ", end-start, end="\n\n")
