import numpy as np

class popularity_RS():
    def __init__(self, R):
        counts = np.sum(R, axis=0)
        counts -= min(counts)
        self.probs = counts/sum(counts)
        print("created Pop RS")

    def get_recommendation(self, user_index, movie_indexs, removed_movie):
        return self.probs[movie_indexs]
