# This file includes all necessary data preprocessing functions for the project.
from itertools import chain
import json
from ordered_set import OrderedSet
import os
import pickle
from scipy.sparse import save_npz, coo_matrix

def dump_all_songs(filename, data_loc):
    '''
    Input:
        - filename: name of file to dump
        - data_loc: path to data source
    This function creates a file containing all the playlists in our dataset.
    Output: List of lists containing all songs.
    '''
    all_songs = [] # all songs
    files = os.listdir(data_loc) # store source data location
    for file in files:
        # load each file
        print(f"Getting {file}...")
        data = json.load(open(data_loc + file))
        # list of playlist are under the key "playlists"
        for row in data["playlists"]:
            # create tracks list
            tracks_in_each_playlist = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
            all_songs.append(tracks_in_each_playlist) # store all songs

    with open(filename, 'wb') as f:
        # save all tracks
        pickle.dump(all_songs, f, pickle.HIGHEST_PROTOCOL)
    print("Full playlist containing all songs created successfully.")
    return all_songs

def create_playlists(filename_MF, filename_leftout, data_loc, reduced_percentage = 2):
    '''
    Input:
        - filename_MF: file which contains playlists for matrix factorization
        - filename_leftout: file which contains leftout songs of each playlists
        - data_loc: path to source file
        - reduced_percentage: percent of playlists you want to consider; Default: 2
    Output: 1) Playlist containing only those songs which serve as input to the Matrix Factorization (MF).
            2) Playlist containing only those songs which were left out and are no input to the MF
    '''

    MF_songs = [] # songs for MF only
    leftout_songs = [] # all songs which are left out of each playlist; list of lists
    # get all files (dataset is divided)
    files = os.listdir(data_loc) # store source data location
    amount_files = len(files)
    # consider given percentage of files
    limit = int(amount_files * (reduced_percentage/100))
    reduced_files = files[:limit]
    del files
    for file in reduced_files:
        print(f"Getting {file}...")
        data = json.load(open(data_loc + file))
        # list of playlist are under the key "playlists"
        for row in data["playlists"]:
            # define percentage of songs which will be used for MF in each playlist
            leftout_cutoff = int(round(len(row) * .9, 0))  # 90% of each playlist for creating recommendation
            # create tracks list
            tracks_in_each_playlist = [song["artist_name"] + " - " + song["track_name"] for song in row["tracks"]]
            # append 90% to tracks list used for MF
            MF_songs.append(tracks_in_each_playlist[:leftout_cutoff]) # first 90%
            # append 10% of songs in each playlist to leftout_songs for evaluation
            leftout_songs.append(tracks_in_each_playlist[leftout_cutoff:]) # last 10% for evaluation

    with open(filename_MF, 'wb') as f:
        # save playlists for MF
        pickle.dump(MF_songs, f, pickle.HIGHEST_PROTOCOL)

    with open(filename_leftout, 'wb') as f:
        # save left out songs for evaluation
        pickle.dump(leftout_songs, f, pickle.HIGHEST_PROTOCOL)
    print("Playlists successfully saved as .pickle files.")

    return MF_songs, leftout_songs


def create_song_map(filename, playlist_all_songs, percentage = 100):
    '''
    Input:
        - filename: file which the songmap should be dumped to
        - playlists_all_songs: playlist file which contains all songs
        - precentage: for how many percent of playlists do you want to create the songmap? Default: 100%
    This function creates a song map for a given percentage of playlists. Default = 100%.
    Output: dictionary {song: index}
    '''
    print("Creating song list...")
    # define cutoff of given percentage
    # only create songmap for playlists until cutoff
    cutoff = int(round(len(playlist_all_songs)*percentage/100,0))
    # create a songlist containing each song; remove duplicates; preserve original order
    song_list = list(OrderedSet(chain(*(playlist_all_songs[:cutoff]))))
    print(f"Length of song_list: {len(song_list)}")
    # create song_map to map each song to its index
    print("Creating song map...")
    song_map = {song: i for i, song in enumerate(song_list)}
    # Save song map
    with open(filename, 'wb') as f:
        pickle.dump(song_map, f, pickle.HIGHEST_PROTOCOL)
    return song_map

def create_sparse_matrix(filename, playlists, song_map):
    '''
    Input:
        - filename: .npz filename of output sparse matrix
        - playlists: all playlists for which to create matrix
        - song_map: song_map mapping each song in given playlists to its index
    This function creates a coordinates sparse matrix with given playlists and corresponding songmap.
    Make sure playlists and song_map correspond!
    Output: coo.matrix
    '''

    # create vectors for sparse matrix
    # data contains only 1s, row_ix contains playlist indices, col_ix contains song indices
    data_m, row_ix, col_ix = [], [], []
    print("Creating vectors...")
    # access each playlists with an index
    for rowIdx, rw in enumerate(playlists):
        # for each playlist
        for dbSong in rw:
            # for each song in a playlist
            row_ix.append(rowIdx)  # row index
            col_ix.append(song_map[dbSong])  # song index
    data_m = [1] * len(col_ix) # create data of same length as number of songs

    # %% Create Sparse Matrix
    print("Creating Sparse Matrix...")
    sparseMatrix = coo_matrix((data_m, (row_ix, col_ix)), shape=(len(playlists), len(song_map)))
    save_npz(filename, sparseMatrix)
    return sparseMatrix