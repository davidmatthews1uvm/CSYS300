import numpy as np
import numba
from tqdm import tqdm 

import av
import matplotlib.pyplot as plt
import argparse


from joblib import Parallel, delayed 


@numba.jit
def candidate_tree_cost(tree_index, world, ag_id_sizes, ag_id_prob, cost, spark_probs):
    """
        computes the new cost of the system if a tree was placed at tree_index.
    """
    new_forest_size = 1 # one tree
    new_burn_prob = spark_probs[tree_index[0], tree_index[1]] # one tree burn prob
    cost_correction_factor = 0
    
    neighbors = [0]*4
    for i in [0, 1]:
        for j in [-1, 1]:
            if (0 <= tree_index[i] + j < world.shape[i]):
                tmp = tree_index.copy()
                tmp[i] += j
                neighbor_tree_id = world[tmp[0], tmp[1]]
                if (neighbor_tree_id not in neighbors):
                    neighbors[i*2+(j+1)//2] = neighbor_tree_id
                    neighbor_size = ag_id_sizes[neighbor_tree_id]
                    neighbor_burn_prob = ag_id_prob[neighbor_tree_id]
                    new_forest_size += neighbor_size
                    new_burn_prob += neighbor_burn_prob
                    cost_correction_factor += neighbor_size * neighbor_burn_prob
    return cost + new_forest_size * new_burn_prob - cost_correction_factor, neighbors
    # TODO: look at how new candidate cost is being calculated. there might be a bug in it!


l_b = 1/10
def build_forest(L, D, l_b=1/10, container=None, stream=None):
    l = L*l_b
    spark_probs = np.fromfunction(lambda i, j: np.exp(-(i+j+2)/l), shape=(L,L), dtype=np.float64)
    spark_probs /= np.sum(spark_probs) # normalize the probability.

    world = np.zeros((L, L), dtype=np.int32) # keep track of the trees in the world.
    ag_id_sizes = np.zeros((L*L), dtype=np.int32) # keep track of each forest size
    ag_id_prob = np.zeros((L*L), dtype=np.float64) # keep track of each forest burn probability
    next_forest_id = 1

    cost = 0
    old_cost = cost


    for iteration_num in tqdm(range(L*L)):
        if container is not None and stream is not None:
            if iteration_num%FRAME_SUBSAMPLE == 0:
                img = (world!=0).astype(np.uint8).reshape((*world.shape, 1))*255
                img = np.repeat(img, TILE_SIZE, axis=0)
                img = np.repeat(img, TILE_SIZE, axis=1)
                img = np.repeat(img, 3, axis=2)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                
                for packet in stream.encode(frame):
                    container.mux(packet)
        # plt.imshow()
        # # plt.show()
        # plt.savefig("figs/L{:d}D{:d}_iter_{:d}.png".format(L, L*L, iteration_num))

        # print(world, tree_id)
        # print(iter)
        empty_spots = np.argwhere(world==0)
        if empty_spots.shape[0] < D:
            trees_to_test = np.arange(empty_spots.shape[0])
        else:
            trees_to_test = np.random.choice(empty_spots.shape[0], D, replace=False)
        
        min_new_cost = 1e100
        min_new_spot = None
        min_spot_neighbors = None
        for t_id in trees_to_test:
            new_cost, neighbors = candidate_tree_cost(empty_spots[t_id], world, ag_id_sizes, ag_id_prob, cost, spark_probs)
            # x, y  = empty_spots[t_id]
            # if ((x == L-3 and y == L-1) or (y == L-3 and x == L-1) or (x == L-2 and y == L-2)):
            #     print(x, y, new_cost)
            # if (iteration_num == 202):
            #     print(t_id, empty_spots[t_id], new_cost, neighbors)
            if (new_cost < min_new_cost):
                min_new_spot = t_id
                min_spot_neighbors = neighbors
                min_new_cost = new_cost
        if (min_new_cost  > cost + 1):
            print(old_cost, min_new_cost, cost, iteration_num, neighbors, min_new_spot, empty_spots[min_new_spot])
            # if we have not already saved the current world to the video, do so now.
            if iteration_num%FRAME_SUBSAMPLE != 0:
                img = (world!=0).astype(np.uint8).reshape((*world.shape, 1))*255
                img = np.repeat(img, TILE_SIZE, axis=0)
                img = np.repeat(img, TILE_SIZE, axis=1)
                img = np.repeat(img, 3, axis=2)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            return (cost, world, ag_id_sizes, ag_id_prob)
        
        old_cost = min_new_cost
        # print(min_new_cost, empty_spots[min_new_spot], min_spot_neighbors )
        forest_neighbors = sorted(set(min_spot_neighbors) - {0}, reverse=True)
        
        x, y  = empty_spots[min_new_spot]
        if len(forest_neighbors) > 0:
            # print(world[x-1:x+2, y-1:y+2])
            # print(forest_neighbors, min_spot_neighbors)
            # print("MERGING FORESTS!!!")
            # print("MERGING FORESTS!!!")
            # print("MERGING FORESTS!!!")
            min_forest_id = forest_neighbors[-1]
            # print(min_forest_id)
            # print(forest_neighbors[:-1])
            world[x, y] = min_forest_id
            for forest_id in forest_neighbors[:-1]:
                # print(world)
                # print(ag_id_sizes[0:next_forest_id+5])
                # print(ag_id_prob[0:next_forest_id+5])
                # print(forest_id, "->", min_forest_id)
                # update the world
                world[world == forest_id] = min_forest_id
                world[world > forest_id] -= 1
                # print(world)

                # update the forest sizes
                ag_id_sizes[min_forest_id] += ag_id_sizes[forest_id] #copy forest size
                ag_id_sizes[forest_id:next_forest_id-1] = ag_id_sizes[forest_id+1:next_forest_id] # update forest name storage
                ag_id_sizes[next_forest_id-1] = 0

                # update the burn probabilities
                ag_id_prob[min_forest_id] += ag_id_prob[forest_id] #copy forest burn prob
                ag_id_prob[forest_id:next_forest_id-1] = ag_id_prob[forest_id+1:next_forest_id] # update forest name storage
                ag_id_prob[next_forest_id-1] = 0
                
                # print(world)
                # print(ag_id_sizes[0:next_forest_id+5])
                # print(ag_id_prob[0:next_forest_id+5])

                # decrement the next forest id number
                next_forest_id -= 1 
            # MAX_MERGES -= 1
            # if MAX_MERGES < 0:
            #     print(world)
            #     print(ag_id_sizes)
            #     print(ag_id_prob)
            #     break
        else:
            # create a new forest.
            # print("created a forest")
            x, y  = empty_spots[min_new_spot]
            world[x, y] = next_forest_id
            ag_id_sizes[next_forest_id] = 1
            ag_id_prob[next_forest_id] = spark_probs[x, y]
            next_forest_id += 1
        cost = np.sum(ag_id_sizes * ag_id_prob)
        # if ( abs(cost-old_cost)/cost > 1e-16):
        # print(old_cost, cost, abs(cost-old_cost)/cost)
            # return ( cost, world, ag_id_sizes, ag_id_prob)
    return (cost, world, ag_id_sizes, ag_id_prob)

global N, TILE_SIZE, FRAME_SUBSAMPLE

FPS = 60
TARGET_DURATION = 10

STREAM_SIZE = 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="System Size", default=32, type=int)
    args = parser.parse_args()
    N = args.N
    TILE_SIZE = STREAM_SIZE // N
    assert STREAM_SIZE == TILE_SIZE * N

    FRAME_SUBSAMPLE = max(int((N**2 )/(FPS * TARGET_DURATION)), 1)
    
    container = av.open('L{:d}D{:d}.mp4'.format(N, N**2), mode='w')
    stream = container.add_stream('mpeg4', rate=FPS)
    stream.width = STREAM_SIZE
    stream.height = STREAM_SIZE
    stream.pix_fmt = 'yuv420p'

    cost, world, ag_id_sizes, ag_id_prob = build_forest(N, N**2, container=container, stream=stream)
    plt.imshow(world!=0)
    plt.savefig("L{:d}D{:d}.png".format(N, N**2))
    print(np.sum((world!=0)) - cost)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()
