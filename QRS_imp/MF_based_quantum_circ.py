import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from QRS_imp.basic_QRS import basic_QRS
from pennylane import numpy as np
import time
from utils_func import export_data, import_data
import visualiser
from QRS_imp.QRS_samples_circs import QRS_infinite_circ, QRS_num_of_items_circ

from multiprocessing import Pool
import math

n_wires = 7
wires_list = list(range(n_wires))

PI = 3.14
hPI = 1.57


dev_tiny_quantum_circ = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_tiny_quantum_circ)
def tiny_quantum_circ(params):
    for wire in wires_list:
        qml.Hadamard(wire)
    for p in params:
        StronglyEntanglingLayers(p, wires=wires_list)

    return qml.probs(wires=wires_list)




# Model Arch: layer of embedded user parasm (rotating only in Z)
# followed by item params, training only the item params one after the other.
class QRS_MF(basic_QRS):
    def __init__(self, R, embedded_ves=[], params_layers=5):
        basic_QRS.__init__(self, R)
        print(R.shape)
        self.embedded_vecs = embedded_ves
        self.params_layers = params_layers
        self.params = self.randomize_init_params_QRS(self.params_layers)
        self.max_opt_layer = 100
        self.history_removal_expected_vec = []
        self.hist_removal_params = []
        for user in range(self.users_num):
            self.hist_removal_params.append(self.randomize_init_params_QRS(5))

        self.infinite_probs = np.zeros(self.users_num)

        self.num_of_item_probs = np.zeros(self.users_num)
        self.num_of_item_div2_probs = np.zeros(self.users_num)
        self.num_of_item_sqrt_probs = np.zeros(self.users_num)
        self.num_of_item_log2m_probs = np.zeros(self.users_num)

    # calculating the cost on the basic_QRS_cric
    # getting as inputs:  *params = pointer to list of parameters and user=user, item=item expected_probs=expected_probs
    # params should be tensonr : [tensor([ [[X ,  Y,  Z](qubit0),
    #                                       [X ,  Y,  Z](qubit1),
    #                                       [X ,  Y,  Z](qubit2) ] (layer0) ]
    #                                      , requires_grad=True)]
    # dim is tensor( 1 x LAYER x QUBITS x 3, requires_grad )
    # returns optimized params according to requires_grad field
    # example to a call:
    # params = opt_item_item.step(self.total_cost_basic_QRS_user_items, *params, user=user, item=item, expected_probs=expected_probs)
    def calc_QRS_loss(self, *params, **kwargs):
        user = kwargs['user']
        probs = QRS_infinite_circ(params)
        if self.hist_removal_en == 0:
            deltas = self.expected_probs_vecs[user] - probs
            deltas = deltas[self.interacted_items_matrix[user]]
            return sum(deltas**2)
        else:
            deltas = self.history_removal_expected_vec[user] - probs
            return sum(deltas ** 2)

        # return sum(((kwargs['expected_probs'] - probs) ** 2))


    def construct_param_list(self, user, layer_requires_grad=False):
        params_list = []
        embedded_vecs = np.array(self.embedded_vecs[user])
        embedded_vecs.requires_grad = False
        for layer_num, layer in enumerate(self.params):
            req_grad = layer_requires_grad
            if layer_num >= self.max_opt_layer: break
            params_list.append(embedded_vecs)
            params_list.append(np.array([layer], requires_grad=req_grad))
        return params_list



    def update_param_list(self, params):
        for i, p in enumerate(params[1::2]):
            self.params[i] = p[0]


    def append_hist_params(self, params, user):
        for p in self.hist_removal_params[user]:
            params.append(np.array([p], requires_grad=True))


    def update_hist_param(self, params, user):
        for i,p in enumerate(params[-1*len(self.hist_removal_params[user]):]):
            self.hist_removal_params[user][i] = p[0]





    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    # each process trains single item - it takes all its users (whom interacted with the item)
    # and calc the loss for that circ
    def train(self, num_of_epochs, LR, export=0, max_opt_layer=100):
        print("\n------- TRAINING RECOMMENDATION SYS -------")
        self.max_opt_layer = max_opt_layer

        opt_item_item = qml.AdagradOptimizer(stepsize=LR, eps=1e-08) #AdamOptimizer(stepsize=0.05, beta1=0.9, beta2=0.999, eps=1e-08)
        for epoch in range(num_of_epochs):
            epoch_start_t = time.time()
            params = self.construct_param_list(0, True)
            for user in range(self.users_num):
                print("epoch:", epoch, "user: ", user, "/", self.users_num)
                for i in range(len(params))[::2]:
                    params[i] = np.array(self.embedded_vecs[user], requires_grad=False)
                for i in range(5):
                    params = opt_item_item.step(self.calc_QRS_loss, *params, user=user)
            self.update_param_list(params)
            if export == 1:
                export_data(self.params, "qrs_mf_params")
                export_data(self.max_opt_layer, "qrs_max_opt_layer")
            print("epoch took:", int(time.time() - epoch_start_t), "secs")






    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    # each process trains single item - it takes all its users (whom interacted with the item)
    # and calc the loss for that circ
    def train_hist_removal(self):
        self.hist_removal_en = 1
        print("\n------- TRAINING RECOMMENDATION SYS -------")
        self.get_reco_matrix()
        for user in range(self.users_num):
            print("user: ", user, "/", self.users_num)
            opt_item_item = qml.AdagradOptimizer(stepsize=0.1,
                                                 eps=1e-08)  # AdamOptimizer(stepsize=0.05, beta1=0.9, beta2=0.999, eps=1e-08)
            params = self.construct_param_list(user, False)
            self.append_hist_params(params, user)
            for i in range(10):
                params = opt_item_item.step(self.calc_QRS_loss, *params, user=user)
            self.update_hist_param(params, user)

        export_data(self.hist_removal_params, "hist_removal_params")
        self.hist_removal_en = 1



    def load_params(self, LOAD_DIR):
        self.params = import_data(LOAD_DIR, "qrs_mf_params")
        # self.max_opt_layer = import_data(LOAD_DIR, "qrs_max_opt_layer")
        self.max_opt_layer = 5

    def load_hist_removal_params(self, LOAD_DIR):
        self.hist_removal_params = import_data(LOAD_DIR, "hist_removal_params")
        self.hist_removal_en = 1



    def get_recommendation(self, user, uninter_movies, removed_movie):
        params = self.construct_param_list(user)
        if self.hist_removal_en:
            self.append_hist_params(params, user)
        probs = tiny_quantum_circ(params)

        # performing history removal if not done yet
        if self.hist_removal_en == 0:
            probs[self.interacted_items_matrix[user]] = 0
            probs[self.bad_interacted_items_matrix[user]] = 0
            probs = probs/sum(probs)

        for i,v in enumerate(probs):
            if i not in uninter_movies:
                probs[i] = int(0)
        probs /= sum(probs)

        visualiser.print_colored_matrix(probs, [[], self.interacted_items_matrix[user], [removed_movie]], 1,1,3)
        return probs

    def get_reco_matrix(self):
        for user in range(self.users_num):
            params = self.construct_param_list(user)
            probs = QRS_infinite_circ(params)
            probs[self.interacted_items_matrix[user]] = 0
            probs[self.bad_interacted_items_matrix[user]] = 0
            self.history_removal_expected_vec.append(probs/sum(probs))



    def get_recommendation2(self, user, uninteracted_items, removed_movie):

        if self.QRS_recommendation_circ == "infinite":
            probs = self.infinite_probs[user]
        if self.QRS_recommendation_circ == "num_of_item":
            probs = self.num_of_item_probs[user]
        if self.QRS_recommendation_circ == "num_of_item_div2":
            probs = self.num_of_item_div2_probs[user]
        if self.QRS_recommendation_circ == "num_of_item_sqrt":
            probs = self.num_of_item_sqrt_probs[user]
        if self.QRS_recommendation_circ == "num_of_item_log2m":
            probs = self.num_of_item_log2m_probs[user]

        # DEBUG
        print("recommendation for user:", user)
        interacted_items = self.interacted_items_matrix[user]
        bad_interacted_items = self.bad_interacted_items_matrix[user]
        visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([removed_movie])],
                                        is_vec=1,
                                        all_positive=1, digits_after_point=4)
        return probs


    def collect_QRS_reco_data(self):
        for user in range(self.users_num):
            params = self.construct_param_list(user)
            if self.hist_removal_en: self.append_hist_params(params, user)

            self.infinite_probs[user] = QRS_infinite_circ(params)
            samples = QRS_num_of_items_circ(params)

            num_of_item_return_probs = np.zeros(self.items_num)
            num_of_item_div2_return_probs = np.zeros(self.items_num)
            num_of_item_sqrt_return_probs = np.zeros(self.items_num)
            num_of_item_log2m_return_probs = np.zeros(self.items_num)

            num_of_item_samples=int(self.items_num)
            num_of_item_div2_samples=int(self.items_num/2)
            num_of_item_sqrt_samples=int(math.ceil(math.sqrt(self.items_num)))
            num_of_item_log2m_samples=int(math.ceil(math.log2(self.items_num)))

            for sample in samples:
                index = int("".join(str(i) for i in sample), 2)
                if num_of_item_samples:
                    num_of_item_return_probs[index] += 1
                    num_of_item_samples-=1
                if num_of_item_div2_samples:
                    num_of_item_div2_return_probs[index] += 1
                    num_of_item_div2_samples -= 1
                else:
                    continue
                if num_of_item_sqrt_samples:
                    num_of_item_sqrt_return_probs[index] += 1
                    num_of_item_sqrt_samples -= 1
                else:
                    continue
                if num_of_item_log2m_samples:
                    num_of_item_log2m_return_probs[index] += 1
                    num_of_item_log2m_samples -= 1
                else:
                    continue
            self.num_of_item_probs[user] = num_of_item_return_probs
            self.num_of_item_div2_probs[user] = num_of_item_div2_return_probs
            self.num_of_item_sqrt_probs[user] = num_of_item_sqrt_return_probs
            self.num_of_item_log2m_probs[user] = num_of_item_log2m_return_probs


