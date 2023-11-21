import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane import numpy as np
from multiprocessing import Pool

from QRS_imp.basic_QRS import basic_QRS
from QRS_imp.QRS_samples_circs import QRS_infinite_circ, QRS_num_of_items_circ
import visualiser
from utils_func import import_data, export_data, run_time
import time
import math




# Model Arch: layer of embedded user parasm (rotating only in Z)
# followed by item params, training only the item params one after the other.
class QRS_items_circs_parallel(basic_QRS):
    def __init__(self, R, item_layers):
        basic_QRS.__init__(self, R)
        self.history_removal_trained = False

        self.infinite_probs = {}
        self.num_of_item_probs = {}
        self.num_of_item_div2_probs = {}
        self.num_of_item_sqrt_probs = {}
        self.num_of_item_log2m_probs = {}


        self.item_params_layers = item_layers
        self.item_params = {}
        for item in range(self.items_num):
            self.item_params[item] = self.randomize_init_params_QRS(self.item_params_layers)

        self.history_params_layers = 2
        self.hist_removal_params = {}
        self.QRS_recommendation_circ = "infinite"
        for user in range(self.users_num):
            self.hist_removal_params[user] = {}
            for item in self.interacted_items_matrix[user]:
                self.hist_removal_params[user][item] = self.randomize_init_params_QRS(self.history_params_layers)




    # calculating the cost on the basic_QRS_cric
    # getting as inputs:  *params = pointer to list of parameters and user=user, item=item expected_probs=expected_probs
    # returns optimized params according to requires_grad field
    # example to a call:
    # params = opt_item_item.step(self.total_cost_basic_QRS_user_items, *params, user=user, item=item, expected_probs=expected_probs)
    def calc_QRS_loss(self, *params, **kwargs):
        probs = QRS_infinite_circ(params)
        # print("loss:",sum(((kwargs['expected_probs'] - probs) ** 2)*(kwargs['cost_ampl_mask'])))
        # return sum(((kwargs['expected_probs'] - probs) ** 2)*(kwargs['cost_ampl_mask']))
        return sum(((kwargs['expected_probs'] - probs) ** 2))




    # ----------------------------------------------------- TRAIN ------------------------------------------------------
    # each process trains single item - it takes all its users (whom interacted with the item)
    # and calc the loss for that circ
    def train(self):
        print("\n------- TRAINING RECOMMENDATION SYS -------")

        p = Pool(5)
        work_sets = []

        for item in range(self.items_num):
            if len(self.users_interacted_with_item_matrix[item]) == 0: continue
            work_sets.append((item, self.users_interacted_with_item_matrix[item]))

        updated_params = p.map(self.optimize_item_params, work_sets)
        print("Done Training Item Params")
        for (item, params) in updated_params:
            self.update_params(item, -1, params, 'item')

    def optimize_item_params(self, work_set):
        item, users_interacted_with_item = work_set
        opt_item_item = qml.AdamOptimizer(stepsize=0.15, beta1=0.9, beta2=0.999, eps=1e-08)
        print("Training item:", item, " - ", len(users_interacted_with_item))

        # creating expected probs vector
        expected_probs = np.zeros(self.items_num)
        for i in users_interacted_with_item:
            expected_probs += self.expected_probs_vecs[i]
        expected_probs /= len(users_interacted_with_item)
        # visualiser.print_colored_matrix(expected_probs, [np.array([]), np.array([]), np.array([])],
        #                                 is_vec=1,
        #                                 all_positive=1, digits_after_point=4)

        # create list of 3D tensors - each tensor is set of parameters
        item_params = self.construct_param_list(self.item_params[item], True)

        for epoch in range(25):
            item_params = [opt_item_item.step(
                self.calc_QRS_loss, *item_params,
                expected_probs=expected_probs
                        )]
            # probs = QRS_embedded_layer_item_layer_circ(item_params)  # getting the QRS probs
            # visualiser.print_colored_matrix(probs, [np.array([]), np.array([]), np.array([])],
            #                                 is_vec=1,
            #                                 all_positive=1, digits_after_point=4)
        return item, item_params


    # this is the old optimze function - for each item we are optimizing for ecry item separelty
    # def optimize_item_params2(self, work_set):
    #     item, users_interacted_with_item = work_set
    #     opt_item_item = qml.AdamOptimizer(stepsize=0.15, beta1=0.9, beta2=0.999, eps=1e-08)
    #     print("Training item:", item, " - ", len(users_interacted_with_item))
    #     # create list of 3D tensors - each tensor is set of parameters
    #     item_params = self.construct_param_list(self.item_params[item], True)
    #     prev_probs_expected = self.expected_probs_vecs[users_interacted_with_item[0]]
    #     # cost_ampl_mask = np.ones(self.items_num)
    #     overall_st = time.time()
    #     for epoch1 in range(5):
    #         st = time.time()
    #         for user in users_interacted_with_item:
    #             expected_probs = prev_probs_expected/2 + (1/2)*(self.expected_probs_vecs[user])
    #             print("training user:", user)
    #             interacted_items = self.interacted_items_matrix[user]
    #             bad_interacted_items = self.bad_interacted_items_matrix[user]
    #             visualiser.print_colored_matrix(expected_probs, [bad_interacted_items, interacted_items, np.array([])],
    #                                             is_vec=1,
    #                                             all_positive=1, digits_after_point=4)
    #
    #             for epoch2 in range(25):
    #                 item_params = [opt_item_item.step(
    #                     self.calc_QRS_loss, *item_params, user=user, item=item,
    #                     # cost_ampl_mask = cost_ampl_mask,
    #                     expected_probs=expected_probs
    #                             )]
    #                 probs = QRS_infinite_circ(item_params)  # getting the QRS probs
    #                 visualiser.print_colored_matrix(probs, [bad_interacted_items, interacted_items, np.array([])],
    #                                                 is_vec=1,
    #                                                 all_positive=1, digits_after_point=4)
    #             prev_probs_expected = expected_probs
    #         print("item:", item, "epoch", epoch1, "took:", time.time() - st)
    #     print("Done - Training item:", item, "took:", time.time() - overall_st)
    #     return item, item_params



    # ----------------------------------------------------- TRAIN HIST ------------------------------------------------
    def train_hist_removal(self):
        print("\n------- TRAINING HISTORY REMOVAL -------")
        p = Pool(6)
        work_sets = []
        for user in range(self.users_num):
            for item in self.interacted_items_matrix[user]:
                work_sets.append((user, item))

        updated_params = p.map(self.optimize_hist_rem_params, work_sets)
        print("Done Training Hist Rem Params")
        for (user, item, params) in updated_params:
            self.update_params(item, user, params, 'item', 'hist')
        self.history_removal_trained = True

    def optimize_hist_rem_params(self, work_set):
        user, item = work_set


        print("Training user:", user, "item:", item)
        opt_item_item = qml.AdamOptimizer(stepsize=0.3, beta1=0.9, beta2=0.999, eps=1e-08)

        # calc the expected prob for the ghist removal
        item_params = self.construct_param_list(self.item_params[item], False)
        expected_probs = QRS_infinite_circ(item_params)            # getting the QRS probs
        expected_probs = self.history_removal_expected_vec(expected_probs, user)    # zero the interacted item probs
        expected_probs = self.amplitude_amplification_expected_vec(expected_probs)  # keep that biggest X probs
        # creating list of 3D tensors - each tensor is set of parameters
        item_params = self.construct_param_list(self.item_params[item], False,
                                                self.hist_removal_params[user][item], True)

        for i in range(10):
            item_params = opt_item_item.step(
                self.calc_QRS_loss, *item_params, user=user,
                # cost_ampl_mask=np.ones((self.items_num), requires_grad=False),
                expected_probs=expected_probs)

        # interacted_items = self.interacted_items_matrix[user]
        # bad_interacted_items = self.bad_interacted_items_matrix[user]
        # expected_probs = QRS_embedded_layer_item_layer_circ(item_params)  # getting the QRS probs
        # visualiser.print_colored_matrix(expected_probs, [bad_interacted_items, interacted_items, np.array([])],
        #                                 is_vec=1,
        #                                 all_positive=1, digits_after_point=4)

        return user, item, item_params



    # input: list contains: params1, req_grad1, params2 , req_grad2 ....
    def construct_param_list(self, *params_and_gard):
        params_list = []
        for params, req_grad in zip(params_and_gard[::2], params_and_gard[1::2]):
            t = params.copy()
            t.requires_grad = req_grad
            params_list.append(t)
        return params_list


    def update_params(self, item, user, params_list, *params_type_list):
        for i, params_type in enumerate(params_type_list):
            if params_type == 'item':
                self.item_params[item] = params_list[i]
            if params_type == 'hist':
                self.hist_removal_params[user][item] = params_list[i]



    def get_recommendation(self, user, uninteracted_items, removed_movie):

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


    def collect_QRS_reco_data(self, rec_sets):
        p = Pool(6)
        for user, removed_movie, uninter_movies in rec_sets:

            work_sets = []
            for item in self.interacted_items_matrix[user]:
                work_sets.append([user, item])

            print("collecting reco data for user", user, " - ", len(work_sets))
            # running for every interaction for the user - and sum the histograms
            # reco_overall_probs is [len(work_sets) X 5 X num_of_items]
            reco_overall_probs = p.map(self.parallel_recommendation_for_user, work_sets)
            infinite_reco_overall_probs          = sum(i[0] for i in reco_overall_probs)
            num_of_item_reco_overall_probs       = sum(i[1] for i in reco_overall_probs)
            num_of_item_div2_reco_overall_probs  = sum(i[2] for i in reco_overall_probs)
            num_of_item_sqrt_reco_overall_probs  = sum(i[3] for i in reco_overall_probs)
            num_of_item_log2m_reco_overall_probs = sum(i[4] for i in reco_overall_probs)

            self.infinite_probs[user] = infinite_reco_overall_probs/sum(infinite_reco_overall_probs)
            self.num_of_item_probs[user] = num_of_item_reco_overall_probs / sum(num_of_item_reco_overall_probs)
            self.num_of_item_div2_probs[user] = num_of_item_div2_reco_overall_probs / sum(num_of_item_div2_reco_overall_probs)
            self.num_of_item_sqrt_probs[user] = num_of_item_sqrt_reco_overall_probs / sum(num_of_item_sqrt_reco_overall_probs)
            self.num_of_item_log2m_probs[user] = num_of_item_log2m_reco_overall_probs / sum(num_of_item_log2m_reco_overall_probs)



    def parallel_recommendation_for_user(self, work_set):
        user, item = work_set
        if self.history_removal_trained == False:
            item_params = self.construct_param_list(self.item_params[item], False)
        else:
            item_params = self.construct_param_list(self.item_params[item], False,
                                                    self.hist_removal_params[user][item], False)

        infinite_return_probs = QRS_infinite_circ(item_params)
        samples = QRS_num_of_items_circ(item_params)

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

        return [infinite_return_probs, num_of_item_return_probs, num_of_item_div2_return_probs, num_of_item_sqrt_return_probs, num_of_item_log2m_return_probs]





    def history_removal_expected_vec(self, input_probs, user):
        input_probs[self.interacted_items_matrix[user]] = 0
        input_probs[self.bad_interacted_items_matrix[user]] = 0
        return input_probs / sum(input_probs)

    def amplitude_amplification_expected_vec(self, input_probs):
        t = np.sort(input_probs)
        input_probs[input_probs<t[-10]] = 0
        return input_probs/sum(input_probs)


    def export_model(self, load_rating_data, rating_data_used):
        # tagging the classic data used
        rating_data_used_ex = run_time
        if load_rating_data:
            rating_data_used_ex = rating_data_used
        export_data([self.history_removal_trained, rating_data_used_ex], "QRS_metadata")

        export_data(self.item_params, "QRS_item_params")

        if self.history_removal_trained:
            export_data(self.hist_removal_params, "QRS_hist_removal_params")


    def load_model(self, DIR_NAME, rating_data_used):
        [self.history_removal_trained, rating_data_used_imp] = import_data(DIR_NAME, "QRS_metadata")
        # if rating_data_used_imp != rating_data_used:
        #     print("WARNING - using QRS data different from the rating data it originally created with")
        #     print("use:", rating_data_used_imp)
        #     exit(0)

        self.item_params = import_data(DIR_NAME, "QRS_item_params").tolist()
        if self.history_removal_trained:
            self.hist_removal_params = import_data(DIR_NAME, "QRS_hist_removal_params").tolist()


    # ------------------------------------- THE GRAVEYARD ---------------------------

    # ----------------------------------------------------- TRAIN2 ------------------------------------------------------
    # def train2(self):
    #     opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-08)
    #     print("\n------- TRAINING RECOMMENDATION SYS -------")
    #     p = Pool(6)
    #     for epoch in range(2):
    #         self.total_cost.append(0)
    #         work_sets = []
    #         for user in range(self.users_num):
    #             print("Training user:", user)
    #             self.error_per_user[user].append(0)
    #             for job_num, item in enumerate(self.interacted_items_matrix[user]):
    #                 work_sets.append([user, item, job_num])
    #         self.total_jobs = len(work_sets)
    #         p.map(self.optimize_single_item_circ, work_sets)
    #
    #         print("")
    #         print(f"total cost: {self.total_cost[-1]:.3f}\n")
    #
    #
    #     visualiser.plot_cost_arrs([self.total_cost])
    #     visualiser.plot_cost_arrs(self.error_per_user)


    # def optimize_single_item_circ2(self, work_set):
    #     user, item, job_num = work_set
    #     opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-08)
    #     print("Training user:", user , "item:", item, "-", job_num, "/", self.total_jobs)
    #     # create list of 3D tensors - each tensor is set of parameters
    #     item_params = self.construct_param_list(self.item_params[item], True)
    #     item_params = [opt_item_item.step(
    #         self.total_cost_basic_QRS_user_items, *item_params, user=user, expected_probs=self.expected_probs_vecs[user],
    #         cost_ampl_mask=self.cost_ampl_mask_matrix[user])]
    #     self.update_params(item, user, item_params, 'item')

    # ----------------------------------------------------- TRAIN HIST ------------------------------------------------
    # def train_hist_removal2(self):
    #     self.history_removal_trained = True
    #     print("\n------- TRAINING HISTORY REMOVAL -------")
    #     p = Pool(10)
    #     work_sets = []
    #     for user in range(self.users_num):
    #         for job_num, item in enumerate(self.interacted_items_matrix[user]):
    #             work_sets.append([user, item, job_num])
    #
    #     self.total_jobs = len(work_sets)-1
    #
    #     p.map(self.optimize_single_item_circ_with_hist_rem, work_sets)
    #
    #
    # def optimize_single_item_circ_with_hist_rem2(self, work_set):
    #     user, item, job_num = work_set
    #     print("Training user:", user, "item:", item, "-", job_num, "/", self.total_jobs)
    #
    #     opt_item_item = qml.AdamOptimizer(stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-08)
    #
    #     item_params = self.construct_param_list(self.item_params[item], False)
    #     expected_probs = QRS_embedded_layer_item_layer_circ(item_params)            # getting the QRS probs
    #     expected_probs = self.history_removal_expected_vec(expected_probs, user)    # zero the interacted item probs
    #     expected_probs = self.amplitude_amplification_expected_vec(expected_probs)  # keep that biggest X probs
    #
    #     # creating list of 3D tensors - each tensor is set of parameters
    #     item_params = self.construct_param_list(self.item_params[item], False,
    #                                             self.hist_removal_params[user][item], True)
    #
    #     for i in range(4):
    #         item_params = opt_item_item.step(
    #             self.total_cost_basic_QRS_user_items, *item_params, user=user, expected_probs=expected_probs,
    #             cost_ampl_mask=np.ones((self.items_num), requires_grad=False))
    #     self.update_params(item, user, item_params, 'item', 'hist')
    #
    # input: list of vectors
    # output: list of vectors which all positive, and the sum of each vector is smaller than pi
    # def normalize_embdded_vecotrs(self):
    #     columns_mins = self.user_params.min(axis=0)
    #     self.user_params -= columns_mins     # min in evey columns is 0
    #
    #     global_max = self.user_params.max()+0.0001
    #     self.user_params /= global_max  # max in all data is 1
    #
    #     self.user_params *= (2*math.pi/defines._EMBEDDING_SIZE)       # sum of each row is up to pi
    #
    #     # self.item_embedded_vecs -= self.item_embedded_vecs.min(axis=0)  # min in evey columns is 0
    #     # self.item_embedded_vecs /= (self.item_embedded_vecs.max() + 0.0001)  # max in all data is 1
    #     # self.item_embedded_vecs *= (math.pi / defines._EMBEDDING_SIZE)  # sum of each row is up to pi
