''' Copyright (c) 2017 Roozbeh Sanaei and SUTD-MIT international design centre

Permission is hereby granted,  free of charge,  to any person obtaining a copy
of this software and associated documentation files (the "Software"),  to deal
in the Software without restriction,  including without limitation the rights
to use,  copy,  modify,  merge,  publish,  distribute,  sublicense,  and/or sell
copies of the Software,  and to permit persons to whom the Software is
furnished to do so,  subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS",  WITHOUT WARRANTY OF ANY KIND,  EXPRESS OR
%IMPLIED,  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,  DAMAGES OR OTHER
LIABILITY,  WHETHER IN AN ACTION OF CONTRACT,  TORT OR OTHERWISE,  ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

import csv
import numpy as np
import random
import math
from multiprocessing import Pool
from functools import partial
import sys
from tqdm import trange
import pickle

def cost(DSM,  cluster_matrix,  costs,  pow_cc,objectives):

    v = []
    if 1 in objectives:
        dsm_size = DSM.shape[0]
        io = np.dot(np.dot(cluster_matrix, DSM), cluster_matrix.transpose())
        ioi = io.diagonal()
        ios = np.sum(io)
        iois = np.sum(ioi)
        ioe = ios - iois
        io_extra = ioe * dsm_size
        v.append(io_extra)

    if 2 in objectives:
        cluster_size = np.sum(cluster_matrix, axis=1)
        cscc = np.power(cluster_size, pow_cc)
        io_intra = np.dot(ioi, cscc)
        v.append(io_intra)

    if 3 in objectives:
        dsm_size = DSM.shape[0]
        io = np.dot(np.dot(cluster_matrix, DSM), cluster_matrix.transpose())
        ioi = io.diagonal()
        ios = np.sum(io)
        iois = np.sum(ioi)
        ioe = ios - iois
        io_extra = ioe * dsm_size
        cluster_size = np.sum(cluster_matrix, axis=1)
        cscc = np.power(cluster_size, pow_cc)
        io_intra = np.dot(ioi, cscc)
        total_cost = 0.5*io_intra+0.5*io_extra
        v.append(total_cost)

    if 4 in objectives:
        number_of_modules = len(np.nonzero(cluster_size)[0])
        v.append(number_of_modules)

    return(v)



#remove empty clusters
def trim_clusters(clusters):
    cluster_size = np.sum(clusters, axis=1)
    empty_clusters = np.argwhere(cluster_size == 0)
    clusters = np.delete(clusters, empty_clusters, axis=0)
    cluster_size = np.delete(cluster_size,  empty_clusters)
    return [clusters, cluster_size]

#calculates cluster bid
def Iplus_bid(elmt,  DSM,  Cluster_matrix,  pow_dep,  pow_bid,  cluster_size, constraints, bus_cluster):
    flag = np.zeros(Cluster_matrix.shape[0])
    flag[np.where(np.dot(Cluster_matrix, (1-(constraints[elmt, :]+bus_cluster)))==0)]=1
    exclude_itself = np.identity(DSM.shape[0])
    inputs = np.dot(Cluster_matrix,  DSM[:,  elmt] - exclude_itself[:,  elmt])
    cluster_bid = np.multiply(flag, np.divide(np.power(inputs,  pow_dep), np.power(cluster_size,  pow_bid)));
    return(cluster_bid)


#read a series of float values from a file
def read_floats(file_name):
    file = open(file_name, 'r')
    contents = file.readlines()
    values = [float(value.rstrip('\n')) for value in contents]
    return  values


def rand():
    global rand_index
    global rands
    rand_index = rand_index + 1
    return(rands[rand_index-1])

#compute hash function of cluster matrix
def hash(x):
    return(np.dot(np.sum(x, axis=1), np.array(range(1, (len(x) + 1)))))

class IGTA_core_parameters:
    pow_cc = 1
    pow_bid = -2
    pow_dep = 2
    rand_accept = 0
    rand_bid = 0
    number_of_iterations = 0
    stable_limit = 2


#single thread IGTA# clustering algorithm it takes DSM,
def cluster(DSM, data, bus, constraints, p,core_parameters,objectives):
    dsm_size = DSM.shape[0]
    #pow_cc = 1; pow_bid = -2; pow_dep = 2; rand_accept = 2 * dsm_size; rand_bid = 2 * dsm_size; times = 2; stable_limit = 2
    pow_cc = core_parameters.pow_cc
    pow_bid = core_parameters.pow_bid
    pow_dep = core_parameters.pow_dep
    rand_accept = core_parameters.rand_accept
    rand_bid = core_parameters.rand_bid
    number_of_iterations = core_parameters.number_of_iterations
    stable_limit = core_parameters.stable_limit

    n_clusters = dsm_size

    bus_cluster = np.zeros(dsm_size)
    available_elements = np.ones(dsm_size)
    cluster_matrix = np.identity(dsm_size)

    for i in bus:
        cluster_matrix[i, i] = 0
        bus_cluster[i] = 1

    if (len(bus)>0):
        cluster_matrix[bus[0]]=bus_cluster

    available_elements = (np.nonzero(1-bus_cluster))[0]
    [cluster_matrix, cluster_size] = trim_clusters(cluster_matrix)
    cst = cost(DSM,  cluster_matrix,  data,  pow_cc,objectives)
    total_cost = p*cst[0]+(1-p)*cst[1]


    stable = 0; changed = 0

    while (stable < stable_limit):
        for k in range(number_of_iterations):
            accept1 = 0
            elmt = available_elements[math.floor(random.random()*(len(available_elements)-1))]
            cluster_bid = Iplus_bid(elmt,  DSM,  cluster_matrix,  pow_dep,  pow_bid,  cluster_size, constraints, bus_cluster)
            best_cluster_bid = np.amax(cluster_bid)
            second_best_cluster_bid = np.amax(np.multiply(best_cluster_bid != cluster_bid, cluster_bid))
            if (math.floor(random.random()*rand_bid) == 0):
                best_cluster_bid = second_best_cluster_bid

            if (best_cluster_bid>0):
                affected_list = [i for i in range(len(cluster_bid)) if ((cluster_bid[i]==best_cluster_bid)&(cluster_matrix[i, elmt]==0))]
                n_affected_clusters = len(affected_list)

                if (n_affected_clusters == 0):
                    continue

                my_idx = affected_list[math.floor(random.random()*(len(affected_list)-1))]
               # print(np.sum(cluster_matrix, axis=1))
                new_cluster_matrix = cluster_matrix.copy()
                new_cluster_matrix[:,  elmt] = 0
                new_cluster_matrix[my_idx,  elmt] = 1
                [new_cluster_matrix, new_cluster_size] = trim_clusters(new_cluster_matrix)
                c = cost(DSM, new_cluster_matrix, data, pow_cc,objectives)
        #        print([c[0],c[1],rand_index,hash(new_cluster_matrix)])
        #        if ((c[0] == 116) & (c[1] == 4700)):
        #            print('here')

                new_total_cost = p * c[0] + (1 - p) * c[1]
                if ((new_total_cost <= total_cost) | (math.floor(rand_accept * random.random()) == 0)):
                    accept1 = 1
            if (accept1):
                accept1 = 0
                if (total_cost > new_total_cost):
                    total_cost = new_total_cost.copy()
                    cluster_matrix = new_cluster_matrix.copy()
                    cluster_size = new_cluster_size.copy()
                    changed = True
        if (changed):
            stable = 0
            changed = False
        else:
            stable = stable + 1
    return cluster_matrix




def read_matrix_without_header(file):
    with open(file,  'r') as csvfile:
        file_contents = csv.reader(csvfile,  delimiter=',',  quotechar='"')
        extracted_matrix = [c[1:len(c)] for c in file_contents]
    extracted_matrix.pop(0)
    return (extracted_matrix)

def read_matrix_header(file):
    with open(file,  'r') as csvfile:
        file_contents = csv.reader(csvfile,  delimiter=',',  quotechar='"')
        extracted_header = [c[0] for c in file_contents]
    return (extracted_header)






def print_cluster(c,labels):
    for l in c:
            print([labels[i] for i in list(l.nonzero()[0])])

def iterate_clustering(thread_index,DSM,data,constraints,bus,runs_per_thread,multi_objective,core_parameters,objectives,trade_off_range):

    clusters = []
    if (multi_objective):
        for i in trange(runs_per_thread):
            p = trade_off_range[0]+(trade_off_range[1]-trade_off_range[0])*i/runs_per_thread
            #print(thread_index,i)
            clusters.append(cluster(DSM, data, bus, constraints, p,core_parameters,objectives))
    else:
        p = 0.5
        for i in trange(runs_per_thread):
            #print(thread_index, i)
            clusters.append(cluster(DSM, data, bus, constraints, p,core_parameters,objectives))



    return(clusters)


def find_pareto_points(objective1,objective2,number_of_pareto_points):
    obj_1_min = np.min(objective1)
    obj_1_max = np.max(objective1)
    obj_1_offset = (obj_1_max - obj_1_min) / number_of_pareto_points
    obj_1_min_grid = np.arange(obj_1_min,obj_1_max,obj_1_offset)

    pareto = []
    for i in range(len(obj_1_min_grid)-1):
        indices = (np.where((obj_1_min_grid[i]<=objective1)&(objective1<obj_1_min_grid[i+1])))[0]
        if len(indices)>0:
            pareto.append(indices[np.argmin(objective2[indices])])
    return (pareto)




def read_parameters(parameters_filename):

    # read parameters from parameter file
    parameters = [values[0] for values in read_matrix_without_header(parameters_filename)]
    # load DSM, constraints matrix and data matrix
    DSM = np.array(read_matrix_without_header(parameters[0])).astype(int)

    # set parameters as default values
    constraints = np.ones(DSM.shape)
    data = []
    core_parameters = IGTA_core_parameters()
    core_parameters.rand_accept = 2*DSM.shape[0]
    core_parameters.rand_bid = 2 * DSM.shape[0]
    core_parameters.number_of_iterations = 2 * DSM.shape[0]
    number_of_threads = 24
    runs_per_thread = 50000
    multi_objective = True
    labels = read_matrix_header(parameters[0])[1:]

    #labels = list(range(DSM.shape[0]))



    #if any parameter is given in correct form use that value otherwise leave it as default
    try:
        constraints = np.array(read_matrix_without_header(parameters[1])).astype(int)
        data = np.array(read_matrix_without_header(parameters[2])).astype(float)

        core_parameters.pow_cc = int(parameters[3])
        core_parameters.pow_bid = int(parameters[4])
        core_parameters.pow_dep = int(parameters[5])
        core_parameters.rand_accept = int(parameters[6])
        core_parameters.rand_bid = int(parameters[7])
        core_parameters.number_of_iterations = int(parameters[8])
        core_parameters.stable_limit = int(parameters[9])
        number_of_threads = int(parameters[10])
        runs_per_thread = int(parameters[11])
        if parameters[12] == 'False':
            multi_objective = False
        labels = read_matrix_header(parameters[0])[1:]
    except:
        pass


    return [DSM,constraints,data,core_parameters,number_of_threads,runs_per_thread,multi_objective,labels]


def parallel_clustering(DSM,data,constraints,bus,runs_per_thread,multi_objective,core_parameters,objectives,trade_off_range,number_of_threads):

    if (number_of_threads > 1):
        with Pool(number_of_threads) as pool:
            clustering_results = pool.map(partial(iterate_clustering, DSM=DSM, data=data, constraints=constraints, bus=bus,
                                       runs_per_thread=runs_per_thread, multi_objective=multi_objective,
                                       core_parameters=core_parameters,objectives=objectives,trade_off_range=trade_off_range), range(number_of_threads))
    else:
        clustering_results = [
            iterate_clustering(0, DSM=DSM, data=data, constraints=constraints, bus=bus,
                    runs_per_thread=runs_per_thread,multi_objective=multi_objective,
                    core_parameters=core_parameters,objectives=objectives,trade_off_range=trade_off_range)]

    clustering_matrices = np.concatenate(clustering_results, axis=0)

    return (clustering_matrices)


if __name__ == '__main__':

    compute = True
    # if parameters_filename is given as argument use it otherwise use a default microcontroller file
    try:
        parameters_filename = sys.argv[1]
    except:
        parameters_filename = 'examples/car heater parameters.csv'

        [DSM, constraints, data, core_parameters, number_of_threads, runs_per_thread, multi_objective,labels] = read_parameters(parameters_filename)

    if compute:
        bus = []

        print ("computing clusters ...")

        clustering_matrices1 = parallel_clustering(DSM=DSM, data=data, constraints=constraints, bus=bus,
                                             runs_per_thread=runs_per_thread, multi_objective=multi_objective,
                                             core_parameters=core_parameters, objectives=[1, 2], trade_off_range=[0, 1],number_of_threads=number_of_threads)

        clustering_matrices2 = parallel_clustering(DSM=DSM, data=data, constraints=constraints, bus=bus,
                                               runs_per_thread=runs_per_thread, multi_objective=multi_objective,
                                               core_parameters=core_parameters, objectives=[3, 4],
                                               trade_off_range=[0.999, 1], number_of_threads=number_of_threads)

    #if more than number of threads is more than 1 use a process pool otherwise use a single thread

        clustering_matrices = np.concatenate([clustering_matrices1,clustering_matrices2], axis=0)

        print("computing costs...")
        cluster_costs = np.array([cost(DSM, clustering_matrix, data, 1, [1, 2, 3, 4]) for clustering_matrix in clustering_matrices])

        print("saving information")
        pickle.dump([clustering_matrices, cluster_costs], open('D://temp//save1.p', "wb"))

    else:
        print("loading saved information ...")
        [clustering_matrices, cluster_costs] = pickle.load(open('D://temp//save1.p', "rb"))

    cluster_costs[:, 0]= (cluster_costs[:, 0]-min(cluster_costs[:, 0]))/(max(cluster_costs[:, 0])-min(cluster_costs[:, 0]))
    cluster_costs[:, 1] = (cluster_costs[:, 1] - min(cluster_costs[:, 1])) / (
            max(cluster_costs[:, 1]) - min(cluster_costs[:, 1]))
    cluster_costs[:, 2] = (cluster_costs[:, 2] - min(cluster_costs[:, 2])) / (
            max(cluster_costs[:, 2]) - min(cluster_costs[:, 2]))

    print("computing pareto frontiers ...")
    # find pareto frontier
    pareto1 = find_pareto_points(objective1=cluster_costs[:,0],objective2=cluster_costs[:,1],number_of_pareto_points=300)

    # find pareto frontier
    pareto2 = find_pareto_points(objective1=cluster_costs[:,3], objective2=cluster_costs[:, 2],
                                 number_of_pareto_points=300)




    print("clustering with minimum total cost is:")
    min_index = np.argmin(cluster_costs[:,2])
    print_cluster(clustering_matrices[min_index],labels)
    print('extra cluster cost :', cluster_costs[min_index][0], 'intra cluster cost :', cluster_costs[min_index][1],'total cost:', cluster_costs[min_index][2])
    print('------------------')


    print("clustering with minimum total for different numbers of modules are:")
    for n_modules in list(set(cluster_costs[:,3])):
        print('number of modules:',n_modules)
        n_modules_constrained_indices = n_modules_constrained_indices = np.where(cluster_costs[:,3]==n_modules)[0]
        n_modules_constrained_min_index = n_modules_constrained_indices[np.argmin(cluster_costs[n_modules_constrained_indices, 2])]
        print_cluster(clustering_matrices[n_modules_constrained_min_index],labels)

        print('relative extra cluster cost :', cluster_costs[n_modules_constrained_min_index][0], 'relative intra cluster cost :', cluster_costs[n_modules_constrained_min_index][1],'relative total cost:', cluster_costs[n_modules_constrained_min_index][2])
        print('------------------')

    print("plotting 1st space ...")
    # plot 1st pareto frontier
    if multi_objective:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.scatter(cluster_costs[:,0], cluster_costs[:,1],c = 'black')
        plt.scatter(cluster_costs[pareto1, 0], cluster_costs[pareto1, 1], c=([0,1,0]),marker = "^")
        plt.scatter(cluster_costs[pareto2, 0], cluster_costs[pareto2, 1], c=([1,0,0]),marker = "o")
        matplotlib.rc('xtick', labelsize=16)
        matplotlib.rc('ytick', labelsize=1656)
        #plt.xlabel('Relative intracluster cost')
        #plt.ylabel('Relative extracluster cost')
        plt.show()

    print("plotting 2nd space ...")
    if multi_objective:
        import matplotlib.pyplot as plt
        plt.scatter(cluster_costs[:,3], cluster_costs[:,2],c = 'black')
        plt.scatter(cluster_costs[pareto1, 3], cluster_costs[pareto1, 2], c=([0,1,0]),marker = "^")
        plt.scatter(cluster_costs[pareto2, 3], cluster_costs[pareto2, 2], c=([1,0,0]),marker = "o")
        #plt.xlabel('Number of modules')
        #plt.ylabel('Relative total cost')
        plt.show()