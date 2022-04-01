from scipy import sparse
import pickle
import numpy as np
import time
import json
import os


#%% Environment Variables
filename = "EDA_results.json"

file_songmap = "song_map.pickle"  # full song_map
file_mx_full = "sparseMatrix_full.npz"  # name for full matrix (used in EDA)
file_all_songs = "allSongs.pickle" #all playlists

if not (os.path.isfile(filename)):
    print("Loading files...")

    try:
        mx = sparse.load_npz(file_mx_full)

        # Read song_map to recreate
        with open(file_songmap, 'rb') as f:
            songmap = pickle.load(f)

        with open(file_all_songs, 'rb') as f:
            playlists = pickle.load(f)

        success = True
    except:
        print("Some file(s) not found.")
        success = False

    if success:
        start = time.perf_counter()

        # inverse song_map to easily access songs by index
        inv_map = {v: k for k, v in songmap.items()}

        #%% Popularity
        print("Doing calculations...")
        # calculate column sums; column = song
        colSums = np.asarray(mx.sum(axis=0)) # axis 0 -> column sum; axis 1 -> rows sum
        # save column sum. ind = (row index, column index); cSum = column sum.
        # access song by column index (ind[1]) from inv_map.
        # descending order by column sum
        # output: list of tuples
        popularity_dict = sorted({inv_map[ind[1]]:cSum for ind,cSum in np.ndenumerate(colSums)}.items(),
                                 key=lambda x:x[1], reverse = True)
        print(f"Top 10 most popular songs: {popularity_dict[:10]}")

        # get total number of songs
        # sum up all columns sums
        songs_sum = sum([n for song,n in popularity_dict])
        print(f"Total number of songs (not unique): {songs_sum}")

        # get songs which only appear once
        # all songs where column sum is 1
        unique_songs = [song for song,n in popularity_dict if n == 1]
        print(f"Total number of songs which only appear once: {len(unique_songs)}, i.e. {round(len(unique_songs)/songs_sum*100,1)}%")


        # avg playlist length
        mean_playlist_len = sum(len(pl) for pl in playlists) / len(playlists)
        print(f"Avg. number of songs in a playlist: {mean_playlist_len}")

        # identical playlists
        # Sort the playlists first.
        identical_playlists = 0
        n = 0 # track progress
        total = len(playlists)
        for pl in playlists:
            pl.sort() # sort playlists, otherwise there won't be any match
            n+=1
            print(f"{round(n / total * 100,2)}% done.")

        # convert to np array for faster processing
        playlists = np.array(playlists, dtype=object)
        # store each playlist with its counts
        _, counts = np.unique(playlists, return_counts=True)
        # sum up playlists which appear more than once
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

        # dump results to json file
        with open(filename, "w") as outfile:
            json.dump(results, outfile, indent=4)
    else:
        print("EDA file already exists.")