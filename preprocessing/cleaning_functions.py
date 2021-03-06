# This file includes all necessary data preprocessing functions for the project.
import json
import os
import pickle
from scipy.sparse import save_npz, coo_matrix
from collections import defaultdict
import random


def dump_all_songs(filename, data_loc):
    """
    Input:
        - filename: name of file to dump
        - data_loc: path to data source
    This function creates a file containing all the playlists in our dataset.
    Output: List of lists containing all songs.
    """
    all_songs = []  # all songs
    files = os.listdir(data_loc)  # store source data location
    for file in files:
        # load each file
        print(f"Getting {file}...")
        data = json.load(open(data_loc + file))
        # list of playlist are under the key "playlists"
        for row in data["playlists"]:
            # create tracks list
            tracks_in_each_playlist = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
            all_songs.append(tracks_in_each_playlist)  # store all songs

    with open(filename, 'wb') as f:
        # save all tracks
        pickle.dump(all_songs, f, pickle.HIGHEST_PROTOCOL)
    print("Full playlist containing all songs created successfully.")
    return all_songs


def create_playlists(filename_MF, filename_leftout, data_loc, songmap, reduced_percentage=2, cutoff_factor=.9, min_songs=1):
    """
    Input:
        - filename_MF: file which shall contain playlists for matrix factorization
        - filename_leftout: file which shall contain leftout songs of each playlists
        - data_loc: path to source file
        - songmap: a song mapped to its index within the sparse matrix' columns; used to filter for relevant songs only.
        - reduced_percentage: percent of playlists you want to consider; Default: 2%
        - cutoff_factor: indicates how songs go to the MF; the rest of the songs goes to left_out. Range: 0-1
        - min_songs: minimum number of songs in a playlist.
    Output: 1) Playlist containing only those songs which serve as input to the Matrix Factorization (MF).
            2) Playlist containing only those songs which were left out and are no input to the MF
    """

    MF_songs = []  # songs for MF only
    leftout_songs = []  # all songs which are left out of each playlist; list of lists
    # get all files (dataset is divided)
    files = os.listdir(data_loc)  # store source data location
    amount_files = len(files)
    # consider given percentage of files
    limit = int(amount_files * (reduced_percentage / 100))
    reduced_files = files[:limit]
    del files
    for file in reduced_files:
        print(f"Getting {file}...")
        data = json.load(open(data_loc + file))
        # list of playlist are under the key "playlists"
        playlists = []
        for row in data["playlists"]:
            # create tracks list
            tracks_in_each_playlist = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]
                                       if (song["artist_name"] + " - " + song["track_name"]) in songmap]
            # tracks_in_each_playlist = [playlist for playlist in tracks_in_each_playlist if len(playlist) >= min_songs]
            if len(tracks_in_each_playlist) >= min_songs:
                playlists.append(tracks_in_each_playlist)
            # define percentage of songs which will be used for MF in each playlist
        for playlist in playlists:
            leftout_cutoff = int(round(len(playlist) * cutoff_factor, 0))  # default: 90% of each playlist for creating recommendation
            # default: append 90% to tracks list used for MF

            # take out random samples and use them as leftouts
            leftout = [playlist.pop(random.randrange(len(playlist))) for _ in range(len(playlist) - leftout_cutoff)]

            MF_songs.append(playlist)  # default: first 90%
            # default: append 10% of songs in each playlist to leftout_songs for evaluation
            leftout_songs.append(leftout)  # default: last 10% for evaluation

    with open(filename_MF, 'wb') as f:
        # save playlists for MF
        pickle.dump(MF_songs, f, pickle.HIGHEST_PROTOCOL)

    with open(filename_leftout, 'wb') as f:
        # save left out songs for evaluation
        pickle.dump(leftout_songs, f, pickle.HIGHEST_PROTOCOL)
    print("Playlists successfully saved as .pickle files.")

    return MF_songs, leftout_songs


def create_song_map(filename, playlist_all_songs, percentage=100, min_song_count=1):
    """
    Input:
        - filename: file which the songmap should be dumped to
        - playlists_all_songs: playlist file which contains all songs
        - precentage: for how many percent of playlists do you want to create the songmap? Default: 100%
    This function creates a song map for a given percentage of playlists. Default = 100%.
    Output: dictionary {song: index}
    """
    print("Creating song list...")
    # define cutoff of given percentage
    # only create songmap for playlists until cutoff
    cutoff = int(round(len(playlist_all_songs) * percentage / 100, 0))

    # see how many times each song appears in the dataset
    song_count = {}
    song_count = defaultdict(lambda: 0, song_count)
    for playlist in playlist_all_songs[:cutoff]:
        for song in playlist:
            song_count[song] += 1
    reduced_songs = []
    for song, count in song_count.items():
        if count >= min_song_count:
            reduced_songs.append(song)
    # create a songlist containing each song; remove duplicates; preserve original order
    # song_list = list(OrderedSet(chain(*(playlist_all_songs[:cutoff]))))
    print(f"Length of reduced_songs: {len(reduced_songs)}")
    # create song_map to map each song to its index
    print("Creating song map...")
    song_map = {song: i for i, song in enumerate(reduced_songs)}
    # Save song map
    with open(filename, 'wb') as f:
        pickle.dump(song_map, f, pickle.HIGHEST_PROTOCOL)
    return song_map


def create_sparse_matrix(filename, playlists, song_map):
    """
    Input:
        - filename: .npz filename of output sparse matrix
        - playlists: all playlists for which to create matrix
        - song_map: song_map mapping each song in given playlists to its index
    This function creates a coordinates sparse matrix with given playlists and corresponding songmap.
    Make sure playlists and song_map correspond!
    Output: coo.matrix
    """

    # create vectors for sparse matrix
    # data contains only 1s, row_ix contains playlist indices, col_ix contains song indices
    data_m, row_ix, col_ix = [], [], []
    print("Creating vectors...")
    # access each playlists with an index
    for rowIdx, rw in enumerate(playlists):
        # for each playlist
        for dbSong in rw:
            if dbSong in song_map:
                # for each song in a playlist
                row_ix.append(rowIdx)  # row index
                col_ix.append(song_map[dbSong])  # song index
    data_m = [1] * len(col_ix)  # create data of same length as number of songs

    # Create Sparse Matrix
    print("Creating Sparse Matrix...")
    sparseMatrix = coo_matrix((data_m, (row_ix, col_ix)), shape=(len(playlists), len(song_map)))
    save_npz(filename, sparseMatrix)
    return sparseMatrix
