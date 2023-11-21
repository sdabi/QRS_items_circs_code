import os
import numpy as np
import math
from datetime import date
from datetime import datetime
from torch import save as save_model
from torch import load as load_model
from multiprocessing import Pool
import visualiser
import pandas as pd
import random


today = date.today()
time = datetime.now().strftime("%H:%M:%S")
run_time = (str(today) + "_" + str(time)).replace (":", "_").replace ("-", "_")


# input: reco_hit_index arr - for every recommendation - where the LOO item was
# output: HR@K arr
def create_HRK(hit_arr):
    hits_ind = np.zeros(11)
    for hit in hit_arr:
        if hit <= 10:
            hits_ind[hit] += 1
    _hit_arr = np.cumsum(hits_ind)
    _hit_arr = _hit_arr/len(hit_arr)
    return _hit_arr

# input:    1. list of tuples
#               A. userID - who we removed the interaction from
#               B. moviedId - which was removed from the user
#           2. if num_of_uninter_per_user == 0 - than taking all uninteracted movies
# output:   1. list of triples:
#               A. user index (encoded)
#               B. the movie index which removed (encoded)
#               C. list of uninteracted movies - contains the removed interaction movie (encoded)
def create_recommendation_sets(all_users, all_movies, df, removed_movie_per_uer, num_of_uninter_per_user=0):
    recommendation_sets = []
    for user, removed_movie in zip(all_users, removed_movie_per_uer):
        # getting all uninteracted movies
        uninteracted_movies = np.array(list(
            all_movies - set(df.loc[df["userId"] == user, "movieId"].values.tolist()) - set([removed_movie])))

        # sample X of the uninteracted movies
        if num_of_uninter_per_user != 0:
            uninteracted_movies = np.random.choice(uninteracted_movies, num_of_uninter_per_user, replace=False)

        uninteracted_movies = np.sort(np.append(uninteracted_movies, [removed_movie]))

        recommendation_sets.append((user, removed_movie, uninteracted_movies))
    print("done creating recommendation sets")
    recommendation_sets = sorted(recommendation_sets)
    return recommendation_sets


# input:  uninteracted movies array
#         LOO item
#         scores array - in corelative order to uninteracted movies
# output: the index of the LOO item in the scores array
def get_LOO_index_in_scores_array(uninteracted_movies, LOO_item, reco_scores, search_reco_item_within_all_items=0):
    if search_reco_item_within_all_items == 0:
        t = np.array(uninteracted_movies)
        desired_inter_index_pos = np.where(t == LOO_item)[0][0]
    else:
        desired_inter_index_pos = LOO_item

    # visualiser.print_colored_matrix(reco_scores,
    #                                 [[], [], np.array([desired_inter_index_pos])],
    #                                 is_vec=1,
    #                                 all_positive=1, digits_after_point=4)

    hit_index = (np.where(reco_scores.argsort()[::-1][:len(reco_scores)] == desired_inter_index_pos)[0])[0] + 1
    return hit_index


# input:    1. list of tuples
#               A. userID - who we removed the interaction from
#               B. moviedId - which was removed from the user
#           2. if num_of_uninter_per_user == 0 - than taking all uninteracted movies
# output:   1. list of triples:
#               A. user index (encoded)
#               B. the movie index which removed (encoded)
#               C. list of uninteracted movies - contains the removed interaction movie (encoded)


def TEST_MODEL(MODEL, recommendation_sets, search_reco_item_within_all_items=0):
    reco_hit_index = []

    for user, removed_movie, uninter_movies in recommendation_sets:
        print("search for recommendation for user:", user)
        # get recommendations from the model - on uninter_movies list
        reco_scores = MODEL.get_recommendation(user, uninter_movies, removed_movie)

        # getting the index of the LOO item from the recommendations
        hit_index = get_LOO_index_in_scores_array(uninter_movies, removed_movie, reco_scores,
                                                  search_reco_item_within_all_items)

        reco_hit_index.append(hit_index)

    reco_hit_index = np.array(reco_hit_index)
    HRK = create_HRK(reco_hit_index)
    return HRK


# this function return both HRK when searching for the QRS recommendation within all the items (normal flow)
# also return the HRK when we zero the probs for the items which are not in the uninter_movies for each user
# trying to compare apples to apples
def TEST_QRS_MODEL_WITHIN_SET_OF_UNITER_VS_CLEAN(MODEL, recommendation_sets, search_reco_item_within_all_items=0):
    reco_hit_index = []
    reco_hit_index_only_in_uninter = []

    for user, removed_movie, uninter_movies in recommendation_sets:
        print("search for recommendation for user:", user)
        # get recommendations from the model - on uninter_movies list
        reco_scores = MODEL.get_recommendation(user, uninter_movies, removed_movie)

        # getting the index of the LOO item from the recommendations
        reco_hit_index.append(get_LOO_index_in_scores_array(uninter_movies, removed_movie, reco_scores,
                                                  search_reco_item_within_all_items))

        # zero the probs for the items which arnt in the uninter items list
        items_not_in_uninter_items_set = list(set(list(range(len(reco_scores)))) - set(uninter_movies))
        reco_scores[items_not_in_uninter_items_set] = 0
        reco_scores = reco_scores/sum(reco_scores)

        reco_hit_index_only_in_uninter.append(get_LOO_index_in_scores_array(uninter_movies, removed_movie, reco_scores,
                                                  search_reco_item_within_all_items))


        interacted_items = MODEL.interacted_items_matrix[user]
        bad_interacted_items = MODEL.bad_interacted_items_matrix[user]

        visualiser.print_colored_matrix(reco_scores, [bad_interacted_items, interacted_items, np.array([removed_movie])],
                                        is_vec=1,
                                        all_positive=1, digits_after_point=4)



    reco_hit_index = np.array(reco_hit_index)
    reco_hit_index_only_in_uninter = np.array(reco_hit_index_only_in_uninter)

    HRK = create_HRK(reco_hit_index)
    HRK_uninter_items = create_HRK(reco_hit_index_only_in_uninter)

    return HRK, HRK_uninter_items




def TEST_QRS_MODEL_PERFECT_HIST_REM(MODEL, recommendation_sets, search_reco_item_within_all_items=0):
    reco_hit_index = []
    total_noise = 0
    p = Pool(6)
    for user, removed_movie, uninter_movies in recommendation_sets:
        print("search for recommendation for user:", user)
        # get recommendations from the model - on uninter_movies list
        interacted_items = MODEL.interacted_items_matrix[user]
        bad_interacted_items = MODEL.bad_interacted_items_matrix[user]

        reco_scores = MODEL.get_recommendation(user, uninter_movies, removed_movie, p)
        # getting the index of the LOO item from the recommendations
        zero_probs_ind = list(set(range(len(reco_scores)-1)) - set(uninter_movies))
        reco_scores[zero_probs_ind] = 0
        reco_scores[interacted_items] = 0
        reco_scores[bad_interacted_items] = 0
        reco_scores = reco_scores/sum(reco_scores)

        visualiser.print_colored_matrix(reco_scores, [bad_interacted_items, interacted_items, np.array([removed_movie])],
                                        is_vec=1,
                                        all_positive=1, digits_after_point=4)


        hit_index = get_LOO_index_in_scores_array(uninter_movies, removed_movie, reco_scores,
                                                  search_reco_item_within_all_items)

        reco_hit_index.append(hit_index)

        if search_reco_item_within_all_items:
            total_noise += sum(reco_scores[uninter_movies])
        else:
            total_noise += sum(reco_scores)

    reco_hit_index = np.array(reco_hit_index)
    HRK = create_HRK(reco_hit_index)
    print("---Total noise:", total_noise, "---")
    return HRK



def sample_read_data(df, num_to_sample):
    all_users = list(set(df.userId.tolist()))
    chosen_users = np.random.choice(all_users, len(all_users) - num_to_sample, replace=False)
    print("dropping", len(chosen_users))
    for user in chosen_users:
        df.drop(df.loc[df["userId"] == user].index.tolist(), inplace=True)

    all_movies = list(set(df.movieId.tolist()))
    chosen_movies = np.random.choice(all_movies, len(all_movies) - num_to_sample, replace=False)
    print("dropping", len(chosen_movies))
    for movie in chosen_movies:
        df.drop(df.loc[df["movieId"] == movie].index.tolist(), inplace=True)




def convert_df_to_matrix(users_num, items_num, df):

    # Pivot the DataFrame to get a binary matrix
    mat = df.pivot(index='userId', columns='movieId', values='rating')
    # Fill NaN values with 0 (no rating)
    mat = mat.fillna(0)
    # Convert to numpy array
    mat = mat.to_numpy()
    return mat


def convert_matrix_to_df(inter_mat, uninteracted_rating=0, inter_threshold=0):
    lines_for_df = []
    for user_index, user_ratings in enumerate(inter_mat):
        for item_index, item_rating in enumerate(user_ratings):
            if item_rating == uninteracted_rating: continue

            if   item_rating > inter_threshold: item_rating = 1
            elif item_rating < inter_threshold: item_rating = -1

            timestamp = random.randint(0,1000)
            lines_for_df.append([user_index, item_index, item_rating, timestamp])
    df = pd.DataFrame(lines_for_df, columns=['userId', 'movieId', 'rating', 'timestamp'])
    return df




def export_model(model, name_in):
    print("export path:", run_time)
    p = os.path.join("./exported_data/", run_time)
    if os.path.isdir(p) == 0:
        os.mkdir(p)
    save_model(model.state_dict(), './exported_data/' + run_time + '/' + name_in)

def import_model(model, dir_name, name_in):
    model.load_state_dict(load_model('./exported_data/' + dir_name + '/' + name_in))
    model.eval()


def export_data(data, name_in):
    p = os.path.join("./exported_data/", run_time)
    if os.path.isdir(p) == 0:
        os.mkdir(p)
    with open('./exported_data/' + run_time + '/' + name_in, 'wb') as f:
        np.save(f, data)
    return run_time


def import_data(dir_name, name_in):
    with open('./exported_data/' + dir_name + '/' + name_in, 'rb') as f:
        return np.load(f, allow_pickle=True)
