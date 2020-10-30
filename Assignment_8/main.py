import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.jit
def candidate_tree_cost(tree_index, world, ag_id_sizes, ag_id_prob, cost, spark_probs):
    """
        computes the new cost of the system if a tree was placed at tree_index.
    """
    new_forest_size = 1 # one tree
    new_burn_prob = spark_probs[tree_index[0], tree_index[1]] # one tree burn prob
    cost_correction_factor = 0
    
    for i in [0, 1]:
        for j in [-1, 1]:
            if (0 < tree_index[i] + j < world.shape[i] - 1):
                tmp = tree_index.copy()
                tmp[i] += j
                neighbor_tree_id = world[tmp[0], tmp[1]]
                if (neighbor_tree_id != 0):
                    neighbor_size = ag_id_sizes[neighbor_tree_id]
                    neighbor_burn_prob = ag_id_prob[neighbor_tree_id]
                    new_forest_size += neighbor_size
                    new_burn_prob += neighbor_burn_prob
                    cost_correction_factor += neighbor_size * neighbor_burn_prob
    return new_forest_size * new_burn_prob - cost_correction_factor

def build_forest(L, D, l_b=1/10):
    l = L*l_b
    spark_probs = np.fromfunction(lambda i, j: np.exp(-i/l)*np.exp(-j/l), shape=(L,L))
    spark_probs /= np.sum(spark_probs) # normalize the probability.

    world = np.zeros((L, L), dtype=np.int32) # keep track of the trees in the world.
    ag_id_sizes = np.zeros((L, L), dtype=np.int32) # keep track of each forest size
    ag_id_prob = np.zeros((L, L), dtype=np.int32) # keep track of each forest burn probability
    next_forest_id = 0

    cost = 0

    for tree_id in range(10):
        empty_spots = np.argwhere(world==0)
        if empty_spots.shape[0] < D:
            trees_to_test = np.arange(empty_spots.shape[0])
        else:
            trees_to_test = np.random.choice(empty_spots.shape[0], D, replace=False)
        
        min_new_cost = 1e10
        min_new_spot = None
        for t_id in trees_to_test:
            new_cost = candidate_tree_cost(empty_spots[t_id], world, ag_id_sizes, ag_id_prob, cost, spark_probs)
            if (new_cost < min_new_cost):
                min_new_spot = t_id
                min_new_cost = new_cost
        print(min_new_cost, empty_spots[min_new_spot])
        
        # TODO PLACE TREE!

if __name__ == '__main__':
    build_forest(128, 128*128)