import math

global _BAD_SAMPLED_INTER
_BAD_SAMPLED_INTER = -1

global _REMOVED_INTER
_REMOVED_INTER = 0

global _NUM_OF_USERS
_NUM_OF_USERS = 256 #610

global _NUM_OF_ITEMS
_NUM_OF_ITEMS = 1024 #9724

global _NUM_OF_ITEMS_AMP
_NUM_OF_ITEMS_AMP = 3

global _EMBEDDING_SIZE
_EMBEDDING_SIZE = int(math.log(_NUM_OF_ITEMS, 2))

global _NUM_OF_LAYERS
_NUM_OF_LAYERS = 5*3

global _NUM_OF_QUBITS_TO_OPTIMIZE
_NUM_OF_QUBITS_TO_OPTIMIZE = int(math.log(_NUM_OF_ITEMS, 2))


global _MAX_HIST_INTER_WEIGHT
_MAX_HIST_INTER_WEIGHT = 0.9



