import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from QRS_imp.basic_QRS import basic_QRS
from pennylane import numpy as np

n_wires = 10
wires_list = list(range(n_wires))

PI = 3.14
hPI = 1.57


build_users_to_probs_dict_circ = qml.device('default.qubit', wires=n_wires, shots=200)
@qml.qnode(build_users_to_probs_dict_circ)
def users_to_probs_dict_circ(params):
    for wire in wires_list:
        qml.Hadamard(wire)
    for p in params:
        StronglyEntanglingLayers(p, wires=wires_list)
    return qml.sample()



test_user_circ_n = qml.device('default.qubit', wires=n_wires, shots=20)
@qml.qnode(test_user_circ_n)
def test_user_circ(params):
    for wire in wires_list:
        qml.Hadamard(wire)
    for p in params:
        StronglyEntanglingLayers(p, wires=wires_list)
    return qml.sample()


class QRS():
    def __init__(self, R, embedded_ves=[]):
        self.num_of_users = len(embedded_ves)
        self.embedded_vecs = embedded_ves
        self.R = R
        # eventually this items_users_hits contains how many each item was hitted by user
        # items_users_hits[Item] = [(user0, hit K0 times), (user1, hit K1 times), ... ]
        self.items_users_hits = {}
        self.embedding_layer = []
        for u in range(self.num_of_users):
            self.embedding_layer.append(np.array(embedded_ves[u].tolist()))


    def build_users_to_probs_dict(self):
        for u in range(self.num_of_users):
            print("collecting samples for user:", u)
            samples = users_to_probs_dict_circ([self.embedding_layer[u]])
            for sample in samples:
                index = int("".join(str(i) for i in sample), 2)
                if index not in self.items_users_hits:
                    self.items_users_hits[index] = []
                self.items_users_hits[index].append(u)
        self.create_list_of_tuples_values_and_repetitions()

    # the input is dict which each key is an item
    # the value for every key is list of users indexs hit the item [ user0, user0, user1, user3 ...]
    # the output is same dict but with list of tuples instead of list indexs [(user0, 2), (user1, 1), (user3, 1) .. ]
    # the values in each tuple should consider the wight of a user to this item
    def create_list_of_tuples_values_and_repetitions(self):
        for key in self.items_users_hits:
            values, counts = np.unique(self.items_users_hits[key], return_counts=True)
            tuples = [(v, c) for v, c in zip(values, counts)]
            self.items_users_hits[key] = tuples


    # given test user embedded vecs - running the test circ
    # the output is a list of wights for each user
    def test(self, test_user_vec):
        samples = test_user_circ([test_user_vec])
        total_weights = np.zeros(self.num_of_users)
        for sample in samples:
            index = int("".join(str(i) for i in sample), 2)
            for hit_tuple in self.items_users_hits[index]:
                user, user_weight = hit_tuple
                total_weights[user] += user_weight
        return total_weights


    def load_all_test_users_embedded_vecs(self, train_inter_mat, test_users, test_users_vec):
        self.train_inter_mat = train_inter_mat
        self.test_users_vec = {}
        for i,u in enumerate(test_users_vec):
            self.test_users_vec[test_users[i]] = u

    def get_recommendation(self, user, uninter_movies, removed_movie):
        samples = test_user_circ([self.test_users_vec[user]])
        total_weights = np.zeros(self.num_of_users)
        for sample in samples:
            index = int("".join(str(i) for i in sample), 2)
            if index not in self.items_users_hits:
                continue
            for hit_tuple in self.items_users_hits[index]:
                tuple_user, tuple_user_weight = hit_tuple
                total_weights[tuple_user] += tuple_user_weight

        probs_for_user = np.zeros(len(self.R[0])) # num of items
        for tuple_user, tuple_user_weight in enumerate(total_weights):
            if tuple_user_weight == 0:
                continue
            probs_for_user += np.array(self.train_inter_mat[tuple_user]) * tuple_user_weight
        # min is 0
        probs_for_user -= min(probs_for_user)
        # consider prob only on test vec
        probs_for_user = [probs_for_user[i] for i in uninter_movies]
        # normalizing to sum 1
        probs_for_user = (probs_for_user / sum(probs_for_user))
        return probs_for_user
