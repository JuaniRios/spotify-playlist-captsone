from scipy import sparse
import pickle
import numpy as np
import time
import json
import os


#%% Environment Variables
filename = "EDA_results.json"

if not os.path.isfile(filename):
    #%% Load Data
    # sparse matrix
    print("Loading data...")
    mx = sparse.load_npz("sparse_matrix_full.npz")

    # song map
    with open("../data_preprocessing_python/song_map_full.pickle", 'rb') as f:
        smap = pickle.load(f)

    # playlist
    with open("../data_preprocessing_python/allSongs_full.pickle", 'rb') as f:
        playlists = pickle.load(f)

    start = time.perf_counter()

    # inverse song_map to easily access songs by index
    inv_map = {v: k for k, v in smap.items()}

    #%% Popularity
    print("Doing calculations...")
    colSums = np.asarray(mx.sum(axis=0))
    #print(type(colSums))
    popularity_dict = sorted({inv_map[ind[1]]:cSum for ind,cSum in np.ndenumerate(colSums)}.items(),
                             key=lambda x:x[1], reverse = True)
    print(f"Top 10 most popular songs: {popularity_dict[:10]}")

    # get total number of songs
    songs_sum = sum([n for song,n in popularity_dict])
    print(f"Total number of songs (not unique): {songs_sum}")

    # get songs which only appear once
    unique_songs = [song for song,n in popularity_dict if n == 1]
    print(f"Total number of songs which only appear once: {len(unique_songs)}, i.e. {round(len(unique_songs)/songs_sum*100,1)}%")


    # avg playlist length
    mean_playlist_len = sum(len(pl) for pl in playlists) / len(playlists)
    print(f"Avg. number of songs in a playlist: {mean_playlist_len}")

    # identical playlists
    # Compare each of the 1 million playlist with each of the other 999 999 ones.
    # Sort the playlists first.
    identical_playlists = 0
    n = 0
    total = len(playlists)
    for pl in playlists:
        pl.sort()
        n+=1
        print(f"{n / total * 100}% done.")

    # convert to np array for faster processing
    playlists = np.array(playlists, dtype=object)
    unique, counts = np.unique(playlists, return_counts=True)
    print(f"Total number of duplicate playlists: {sum(counts>1)}")

    # print time taken
    end = time.perf_counter()
    print(f"\n\nTOTAL RUN DURATION: {end-start}")

    #%% Export results
    results = {"Top 10 most popular songs": f"{popularity_dict[:10]}",
               "Total number of songs (not unique)": f"{songs_sum}",
               "Total number of songs which only appear once": f"{len(unique_songs)}",
               "Percentage of unique songs": f"{round(len(unique_songs)/songs_sum*100,1)}",
               "Avg. number of songs in a playlist": f"{mean_playlist_len}",
               "Total number of duplicate playlists": f"{sum(counts>1)}"
               }

    with open(filename, "w") as outfile:
        json.dump(results, outfile, indent=4)
else:
    print("EDA file exists.")