import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from QRS_imp.basic_QRS import basic_QRS
from pennylane import numpy as np
import time
from utils_func import export_data, import_data

n_wires = 10
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
        probs = tiny_quantum_circ(params)
        # print("expected_probs\n", kwargs['expected_probs'])
        # print("probs\n", probs._value)
        return sum(((kwargs['expected_probs'] - probs) ** 2))


    def construct_param_list(self, user, layer_requires_grad=True):
        params_list = []
        embedded_vecs = np.array(self.embedded_vecs[user])
        embedded_vecs.requires_grad = False
        for layer in self.params:
            params_list.append(embedded_vecs)
            params_list.append(np.array([layer], requires_grad=layer_requires_grad))
        return params_list


    def update_param_list(self, params):
        for i, p in enumerate(params[1::2]):
            self.params[i] = p[0]


    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    # each process trains single item - it takes all its users (whom interacted with the item)
    # and calc the loss for that circ
    def train(self, export=0):
        print("\n------- TRAINING RECOMMENDATION SYS -------")
        opt_item_item = qml.AdagradOptimizer(stepsize=0.01, eps=1e-08) #AdamOptimizer(stepsize=0.05, beta1=0.9, beta2=0.999, eps=1e-08)
        for epoch in range(4):
            epoch_start_t = time.time()
            params = self.construct_param_list(0)
            for user in range(self.users_num):
                print("epoch:", epoch, "user: ", user, "/", self.users_num)
                for i in range(len(params))[::2]:
                    params[i] = np.array(self.embedded_vecs[user], requires_grad=False)
                for i in range(5):
                    params = opt_item_item.step(self.calc_QRS_loss, *params, expected_probs=self.expected_probs_vecs[user])
            self.update_param_list(params)
            if export == 1: export_data(self.params, "qrs_mf_params")
            print("epoch took:", int(time.time() - epoch_start_t), "secs")


    def load_params(self, LOAD_DIR):
        self.params = import_data(LOAD_DIR, "qrs_mf_params")


    def get_recommendation(self, user, uninter_movies, removed_movie):
        # if user > 20:
        #     return np.ones(self.items_num)/self.items_num
        params = self.construct_param_list(user, False)
        probs = tiny_quantum_circ(params)
        for i,v in enumerate(probs):
            if i not in uninter_movies:
                probs[i] = int(0)
        probs /= sum(probs)
        print("user", user ," probs:\n", probs,)
        return probs
