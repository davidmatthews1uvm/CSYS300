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
def build_forest(L, D, l_b=1/10, container=None, stream=None, early_termination=True):
    l = L*l_b
    spark_probs = np.fromfunction(lambda i, j: np.exp(-(i+j+2)/l), shape=(L,L), dtype=np.float64)
    spark_probs /= np.sum(spark_probs) # normalize the probability.

    world = np.zeros((L, L), dtype=np.int32) # keep track of the trees in the world.
    best_world = np.zeros((L, L), dtype=np.int32)
    best_world_saved_step_id = 0
    ag_id_sizes = np.zeros((L*L), dtype=np.int32) # keep track of each forest size
    ag_id_sizes_history = np.zeros((L*L, L*L), dtype=np.int32)
    ag_id_prob = np.zeros((L*L), dtype=np.float64) # keep track of each forest burn probability
    next_forest_id = 1

    yield_history = np.zeros((L*L), dtype=np.float64)
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
            x, y  = empty_spots[t_id]
            # if ((x == L-3 and y == L-1) or (y == L-3 and x == L-1) or (x == L-2 and y == L-2)):
            #     print(x, y, new_cost)
            # if (iteration_num == 202):
            #     print(t_id, empty_spots[t_id], new_cost, neighbors)
            if ((new_cost < min_new_cost)): # IF USING MIDDLE TO EDGE TIE BREAKING or (new_cost == min_new_cost and abs(x-y) < abs(empty_spots[min_new_spot][0] - empty_spots[min_new_spot][1]))):
                min_new_spot = t_id
                min_spot_neighbors = neighbors
                min_new_cost = new_cost

        if (best_world_saved_step_id == 0 and min_new_cost  > cost + 1  ):
            best_world[:] = world[:]
            best_world_saved_step_id = iteration_num
            if (early_termination):
                # print(old_cost, min_new_cost, cost, iteration_num, neighbors, min_new_spot, empty_spots[min_new_spot])
                # if we have not already saved the current world to the video, do so now.
                if container is not None and stream is not None and iteration_num%FRAME_SUBSAMPLE != 0:
                    img = (world!=0).astype(np.uint8).reshape((*world.shape, 1))*255
                    img = np.repeat(img, TILE_SIZE, axis=0)
                    img = np.repeat(img, TILE_SIZE, axis=1)
                    img = np.repeat(img, 3, axis=2)
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                return (cost, world, ag_id_sizes, ag_id_prob, yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id)

        old_cost = min_new_cost
        # print(min_new_cost, empty_spots[min_new_spot], min_spot_neighbors )
        forest_neighbors = sorted(set(min_spot_neighbors) - {0}, reverse=True)
        
        x, y  = empty_spots[min_new_spot]
        ag_id_sizes_history[iteration_num] = ag_id_sizes # save the ag id size distribution
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
            ag_id_sizes[min_forest_id] +=1
            ag_id_prob[min_forest_id] += spark_probs[x, y]
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
        yield_history[iteration_num] = 1 + iteration_num - cost
        cost = np.sum(ag_id_sizes * ag_id_prob)
        # if ( abs(cost-old_cost)/cost > 1e-16):
        # print(old_cost, cost, abs(cost-old_cost)/cost)
            # return ( cost, world, ag_id_sizes, ag_id_prob)
    print(next_forest_id )
    return (cost, world, ag_id_sizes, ag_id_prob, yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id)

global N
# global TILE_SIZE, FRAME_SUBSAMPLE

# FPS = 60
# TARGET_DURATION = 20

# STREAM_SIZE = 1024

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--N", help="System Size", default=32, type=int)
    # args = parser.parse_args()
    # for n in range(2, args.N):
    # N = n
    # TILE_SIZE = STREAM_SIZE // N
    # STREAM_SIZE = TILE_SIZE * N

    # FRAME_SUBSAMPLE = max(int((N**2 )/(FPS * TARGET_DURATION)), 1)
    
    # container = av.open('L{:d}D{:d}top.mp4'.format(N, N**2), mode='w')
    # stream = container.add_stream('mpeg4', rate=FPS)
    # stream.width = STREAM_SIZE
    # stream.height = STREAM_SIZE
    # stream.pix_fmt = 'yuv420p'

    N = 128
    data_dict = dict()
    name_dict = {1:"1", 2:"2", N:"L", N*N:"L^2"}
    for D in [1, 2, N, N*N]:
        cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id = build_forest(N, D, early_termination=False) #, container=container, stream=stream)
        data_dict[D] = (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id)
    
    # Part 3a
    fig, axs = plt.subplots(2,2, figsize=(9, 9))
    (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[1]
    print((best_world!=0).sum(), yield_history[best_world_saved_step_id], )
    axs[0,0].imshow(best_world!=0, cmap=plt.cm.gray, interpolation='nearest', aspect='auto')
    axs[0,0].set_title("$D=1$")
    (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[2]
    print((best_world!=0).sum(), yield_history[best_world_saved_step_id])
    axs[0,1].imshow(best_world!=0, cmap=plt.cm.gray, interpolation="none")
    axs[0,1].set_title("$D=2$")
    (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[N]
    print((best_world!=0).sum(), yield_history[best_world_saved_step_id])
    axs[1,0].imshow(best_world!=0, cmap=plt.cm.gray, interpolation="none")
    axs[1,0].set_title("$D=L$")
    (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[N*N]
    print((best_world!=0).sum(), yield_history[best_world_saved_step_id])
    axs[1,1].imshow(best_world!=0, cmap=plt.cm.gray, interpolation="none")
    axs[1,1].set_title("$D=L^2$")
    plt.savefig("Part3a.pdf")

    # Part 3b
    for D in data_dict.keys():
        (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[D]
        plt.plot(np.linspace(0, 1, N*N), yield_history / (N*N), label="$D=%s$"%(name_dict[D]))
        plt.ylabel("Yield")
        plt.xlabel("Trees planted")
        plt.legend()
    plt.savefig("Part3b.pdf")

    # Part 3c
    for D in data_dict.keys():
        (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[D]    
        peak_yield_sizes = ag_id_sizes_history[best_world_saved_step_id]
        peak_yield_sizes = sorted(peak_yield_sizes[peak_yield_sizes != 0], reverse=True)
        plt.plot(np.arange(len(peak_yield_sizes))+1, np.array(peak_yield_sizes), label="D=%d"%(D))
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Number of Trees per forest")
    plt.xlabel("Forest Size Rank")
    plt.legend()
    plt.savefig("Part3c.pdf")

    # Part 3d
    (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[N*N]
    for step_id_to_plot in (np.arange(1, 10) * N*N/10 + 1).astype(np.int):
        peak_yield_sizes = ag_id_sizes_history[step_id_to_plot]
        peak_yield_sizes = sorted(peak_yield_sizes[peak_yield_sizes != 0], reverse=True)
        plt.plot(np.arange(len(peak_yield_sizes))+1, np.array(peak_yield_sizes), label="$\\rho={:.2f}$".format((step_id_to_plot)/(N*N)))
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Number of Trees per forest")
    plt.xlabel("Forest Size Rank")
    plt.legend()
    plt.savefig("Part3d.pdf")

    # Part 3d CDF
    (cost, world, ag_id_sizes, ag_id_prob,  yield_history, ag_id_sizes_history, best_world, best_world_saved_step_id) = data_dict[N*N]
    for step_id_to_plot in (np.arange(1, 10) * N*N/10 + 1).astype(np.int):
        peak_yield_sizes = ag_id_sizes_history[step_id_to_plot]
        peak_size_cdf = sorted(peak_yield_sizes[peak_yield_sizes!=0])
        num_ge_size = range(len(peak_size_cdf), 0, -1)
        plt.plot(peak_size_cdf, num_ge_size,  label="$\\rho={:.2f}$".format((step_id_to_plot)/(N*N)))
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Number of Trees per forest")
    plt.xlabel("Forest Size Rank")
    plt.legend()
    plt.savefig("Part3d_CDF.pdf")

    # plt.imshow(world!=0)
    # plt.savefig("L{:d}D{:d}top.png".format(N, N**2))
    # print(np.sum((world!=0)) - cost)

    # # Flush stream
    # for packet in stream.encode():
    #     container.mux(packet)

    # # Close the file
    # container.close()
