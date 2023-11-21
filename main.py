import numpy as np
import pandas as pd
# import torch

import defines
from QRS_imp.QRS_items_circs_parallel import QRS_items_circs_parallel

from MF.pytourch_MF import MF
from MF.pytourch_MF import train_MF_model
# from MF.pytourch_MF_initial_input import MF
# from MF.pytourch_MF_initial_input import train_MF_model
from utils_func import *

from Other_RS.random_RS import random_RS
from Other_RS.popularity_RS import popularity_RS
import visualiser
import random
import time
from datetime import datetime
import multiprocessing as mp


def load_small_data():
    with open('./data/ml-100k/ml-100k/u.data') as f:
        lines = f.readlines()
        lines = [[eval(a) for a in line.split()] for line in lines]
        df = pd.DataFrame(lines, columns=['userId', 'movieId', 'rating', 'timestamp'])
        return df

def load_jester_data():
    df = pd.read_excel("./data/jester_data/jester-data-3.xls", header=None, names=None, nrows=1000,
                        usecols=list(range(1, 101)))
    print(df)
    inter_mat = df.to_numpy()
    df = convert_matrix_to_df(inter_mat, 99, 0)
    filter_users_by_inters_num(df, 2, 40)
    filter_items_by_inters_num(df, 1, 1000)
    return df



def load_tourism_data():
    df = pd.read_csv("./data/tourism_rating.csv")
    df = df.drop_duplicates(subset=['userId', 'movieId'])
    print(df)
    return df



def filter_users_by_inters_num(df, min_inters_num, max_inters_num):
    users = list(set(df.userId.tolist()))
    count = 0
    for user in users:
        interactions_count = len(df.loc[(df["userId"] == user) & (df["rating"] == 1)])
        if (interactions_count < min_inters_num) or (interactions_count > max_inters_num):
            count += 1
            df.drop(df.loc[df["userId"] == user].index.tolist(), inplace=True)
    print("dropped", count, "users")



def filter_items_by_inters_num(df, min_inters_num, max_inters_num):
    movies = list(set(df.movieId.tolist()))
    count = 0
    for movie in movies:
        if (len(df.loc[(df["movieId"] == movie)]) < min_inters_num) or \
                (len(df.loc[(df["movieId"] == movie)]) > max_inters_num):
            count += 1
            df.drop(df.loc[df["movieId"] == movie].index.tolist(), inplace=True)
    print("dropped", count, "items")



def filter_top_k_users_or_items(df, K, col_to_filter):
    # Filter rows with rating 1
    positive_ratings = df[df['rating'] == 1]
    # Count the number of positive ratings for each user
    counts = positive_ratings[col_to_filter].value_counts()
    # Select the top K
    top_k = counts.head(K).index
    # Filter the original DataFrame based on the top K users
    result_df = df[df[col_to_filter].isin(top_k)]
    return result_df




def remove_last_interactions(df):
    # Sort the DataFrame by user and timestamp in descending order
    df_sorted = df.sort_values(by=['userId', 'timestamp'], ascending=[True, False])

    # Group by user and keep all but the last row for each group
    df_without_last = df_sorted.groupby('userId').apply(lambda x: x.iloc[:-1])

    # Extract the last interaction for each user
    removed_item_ids  = df_sorted.groupby('userId').apply(lambda x: x.iloc[-1]['movieId']).to_numpy()

    # Reset index for the resulting DataFrames
    df_without_last = df_without_last.reset_index(drop=True)

    return df_without_last, removed_item_ids


def remove_single_interaction(df):
    removed_movie_ids = []

    for user_id in df['userId'].unique():
        # Find the indices of interactions for the current user
        user_indices = df[df['userId'] == user_id].index

        # Randomly select one interaction to remove
        removed_index = np.random.choice(user_indices)

        # Append the removed movieId to the list
        removed_movie_ids.append(df.loc[removed_index, 'movieId'])

        # Drop the selected interaction from the DataFrame
        df = df.drop(index=removed_index)

    # Convert the list of removed movieIds to a numpy array
    removed_movie_ids = np.array(removed_movie_ids)

    return df, removed_movie_ids


def proc_col(col):
    name2idx = {o: i for i, o in enumerate(col.unique())}
    return name2idx, np.array([name2idx.get(x, -1) for x in col])


def encode_data(df):
    for col_name in ["userId", "movieId"]:
        _, col = proc_col(df[col_name])
        df[col_name] = col
    return set(df.userId.values.tolist()), set(df.movieId.values.tolist())





""" 
this main - running the following quantum algorithm:
 1. run MF and extract embedded user vecs - and normmalize them
 2. train vqa with layers consisted of embedded vec followed by learning layer
 3. recommendation step: run circ with user embedded vecs   
"""
from QRS_imp.MF_based_quantum_circ import QRS_MF

""" 
this main - embedded vectors followed by learning layer - but on real data   
"""


def thread_train_and_test_QRS(qrs_mf, rec_sets, layers, LR, return_dict):
    HRK_list = []
    HRK_list_to_plot = []
    HRK_QRS = 0
    title = ""
    for epoch in range(5):
        start_time = time.time()
        print("layers:", layers, " epoch:", epoch, " start time:", datetime.now().strftime("%H:%M:%S"))
        qrs_mf.train(num_of_epochs=1, LR=LR, export=0, max_opt_layer=layers)
        HRK_QRS = TEST_MODEL(qrs_mf, rec_sets, 1)
        title = "QRS_" + str(layers) + 'L_' + str(LR) + 'LR_' + str(epoch) + 'E'
        HRK_list_to_plot.append((HRK_QRS, title))
        print(title, " epoch took:", int(time.time() - start_time))

    visualiser.plot_HRK(HRK_list_to_plot)
    HRK_list.append((HRK_QRS, title))
    # qrs_mf.train_hist_removal()
    # HRK_QRS = TEST_MODEL(qrs_mf, rec_sets, 1)
    # title = title+"_HR"
    # HRK_list.append((HRK_QRS, title))
    return_dict[LR] = HRK_list

PI_4 = 0.78
USE_SMALL_DATA = 1
LOAD_DATA = 0
QRS_LOAD_DATA = 0
LOAD_DIR = "2023_10_03_12_18_52_good_128items"
QRS_LOAD_DIR = "2023_10_03_12_18_52_good_128items"




if __name__ == '__main__':
    if LOAD_DATA == 0:

        # --------------- DATA PREPARATION -------------
        df = load_small_data()
        # df = load_tourism_data()

        # df = pd.read_csv("./data/ratings.csv")
        rating_threshold = 0
        df.loc[df["rating"] <= rating_threshold, "rating"] = -1
        df.loc[df["rating"] > rating_threshold, "rating"] = 1

        df = filter_top_k_users_or_items(df, 512, 'userId')
        df = filter_top_k_users_or_items(df, 128, 'movieId')
        filter_users_by_inters_num(df, 1, 32)
        # filter_items_by_inters_num(df,1,1000)

        # df = load_jester_data()

        all_users, all_movies = encode_data(df)
        num_users, num_items = len(all_users), len(all_movies)
        num_of_qubits = math.ceil((math.log(num_items, 2)))
        num_of_embedding_layers = 1
        embed_size = num_of_embedding_layers * num_of_qubits * 3

        print("users:", num_users, "items:", num_items)

        # df, removed_item_per_user = remove_single_interaction(df)
        df, removed_item_per_user = remove_last_interactions(df)

        rec_sets = create_recommendation_sets(all_users, all_movies, df, removed_item_per_user)
        inter_mat = convert_df_to_matrix(num_users, num_items, df)
        print("R shape:", inter_mat.shape)

        # --------------- MF -------------
        MF_model = MF(num_users, num_items, emb_size=embed_size)  # .cuda() if you have a GPU
        user_vecs = train_MF_model(MF_model, df, epochs=100, lr=0.01, lmbd=0.005)

        user_vecs = list(user_vecs.detach().numpy())
        if np.max(user_vecs) > -1 * np.min(user_vecs):
            multiply_factor = PI_4 / np.max(user_vecs)
        else:
            multiply_factor = -1 * PI_4 / np.min(user_vecs)

        for i in range(len(user_vecs)):
            for j in range(len(user_vecs[i])):
                user_vecs[i][j] *= multiply_factor
        user_vecs = [np.reshape(a, (num_of_embedding_layers, num_of_qubits, 3)) for a in user_vecs]

        # export_data(np.array(rec_sets), "rec_sets")
        # export_data([num_users, num_items, embed_size], "metadata")
        # export_data(inter_mat, "inter_mat")
        # export_data(user_vecs, "user_vecs")
        # export_model(MF_model, "MF_model")
    else:
        num_users, num_items, embed_size = import_data(LOAD_DIR, "metadata")
        rec_sets = import_data(LOAD_DIR, "rec_sets")
        inter_mat = import_data(LOAD_DIR, "inter_mat")
        user_vecs = import_data(LOAD_DIR, "user_vecs")
        MF_model = MF(num_users, num_items, emb_size=embed_size)  # .cuda() if you have a GPU
        import_model(MF_model, LOAD_DIR, "MF_model")

    HRK_list = []
    # --------- RANDOM ---------
    RAND_RECO = random_RS()
    HRK_RAND = TEST_MODEL(RAND_RECO, rec_sets)
    HRK_list.append((HRK_RAND, "RAND"))

    # --------- POPULARITY ---------
    POP_RECO = popularity_RS(inter_mat)
    HRK_POP = TEST_MODEL(POP_RECO, rec_sets)
    HRK_list.append((HRK_POP, "POP"))

    # --------- MF ---------
    HRK_MF = TEST_MODEL(MF_model, rec_sets)
    HRK_list.append((HRK_MF, "MF"))
    visualiser.plot_HRK(HRK_list)

    qrs_mf = QRS_MF(inter_mat, user_vecs, 100)

    if QRS_LOAD_DATA == 0:
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        for layers in [15, 25, 35]:
            for LR in [0.005, 0.01, 0.05]:
                p = mp.Process(target=thread_train_and_test_QRS, args=(qrs_mf,rec_sets, layers, LR, return_dict,))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            print(return_dict)
            for layers_res in return_dict.values():
                HRK_list = HRK_list + layers_res
    else:
        qrs_mf.load_params(QRS_LOAD_DIR)
        HRK_QRS = TEST_MODEL(qrs_mf, rec_sets, 1)
        HRK_list.append((HRK_QRS, 'QRS'))

        # qrs_mf.train_hist_removal()

        qrs_mf.get_reco_matrix()
        qrs_mf.load_hist_removal_params(QRS_LOAD_DIR)

        HRK_QRS = TEST_MODEL(qrs_mf, rec_sets, 1)
        HRK_list.append((HRK_QRS, 'QRS_hist_rem'))

    visualiser.plot_HRK(HRK_list)
    print(HRK_list)


