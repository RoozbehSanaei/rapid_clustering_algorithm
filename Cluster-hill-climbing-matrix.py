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

#In this code three different strategies has been used for clustering, clustermatrix and stochastic hill climbing strategy are employed

import csv
import numpy as np
import random
import math
from multiprocessing import Pool
from functools import partial
import sys
from tqdm import trange


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
def Iplus_bid_matrix(  DSM,  Cluster_matrix,  pow_dep,  pow_bid,  cluster_size, constraints, bus_cluster):
    flag = np.zeros(Cluster_matrix.shape)
    flag[np.where(np.dot(Cluster_matrix, np.multiply((1-constraints), np.outer(np.transpose(1-bus_cluster),(1-bus_cluster))))==0)]=1
    exclude_itself = np.identity(DSM.shape[0])
    inputs = np.dot(Cluster_matrix,  DSM - exclude_itself)
    cluster_size_pow = np.power(cluster_size, pow_bid)
    cluster_bid = np.multiply(flag, np.divide(np.power(inputs,  pow_dep), cluster_size_pow[:,None]));
    return(cluster_bid)


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

#move a randomly choosen elment based on cluster bid likelihood to a cluster choosen based based on clusterbid likelihood
def move(available_elements,DSM,cluster_matrix, constraints,pow_dep, pow_bid, cluster_size,bus_cluster,rand_bid):
    new_cluster_matrix = None
    new_cluster_size = None
    modified = False
    cluster_bid = Iplus_bid_matrix( DSM, cluster_matrix, pow_dep, pow_bid, cluster_size, constraints, bus_cluster)+0.00001
    cluster_bid = cluster_bid.flatten()
    prob = np.divide(cluster_bid,(np.sum(cluster_bid)))
    r = random.random()
    pick = np.where(np.add.accumulate(prob)>r)[0][0]
    my_idx = pick//cluster_matrix.shape[1]
    elmt = pick%cluster_matrix.shape[1]
    new_cluster_matrix = cluster_matrix.copy()
    new_cluster_matrix[:, elmt] = 0
    new_cluster_matrix[my_idx, elmt] = 1
    [new_cluster_matrix, new_cluster_size] = trim_clusters(new_cluster_matrix)
    modified = True
    return ([modified,new_cluster_matrix,new_cluster_size])

#move a randomly choosen elment with uniform likelihood to a cluster choosen based on likelihood equal to cluster bid
def move1(available_elements,DSM,cluster_matrix, constraints,pow_dep, pow_bid, cluster_size,bus_cluster,rand_bid):
    new_cluster_matrix = None
    new_cluster_size = None
    elmt = available_elements[math.floor(random.random() * (len(available_elements) - 1))]
    modified = False
    cluster_bid = Iplus_bid(elmt, DSM, cluster_matrix, pow_dep, pow_bid, cluster_size, constraints, bus_cluster)+0.00001
    prob = np.divide(cluster_bid,(np.sum(cluster_bid)))
    my_idx = np.where(np.add.accumulate(prob)>random.random())[0][0]
    new_cluster_matrix = cluster_matrix.copy()
    new_cluster_matrix[:, elmt] = 0
    new_cluster_matrix[my_idx, elmt] = 1
    [new_cluster_matrix, new_cluster_size] = trim_clusters(new_cluster_matrix)
    modified = True
    return ([modified,new_cluster_matrix,new_cluster_size])

#move a randomly choosen elment with uniform likelihood to a cluster choosen based on highest cluster bid
def move(available_elements,DSM,cluster_matrix, constraints,pow_dep, pow_bid, cluster_size,bus_cluster,rand_bid):
    new_cluster_matrix = None
    new_cluster_size = None
    elmt = available_elements[math.floor(random.random() * (len(available_elements) - 1))]
    modified = False
    cluster_bid = Iplus_bid(elmt, DSM, cluster_matrix, pow_dep, pow_bid, cluster_size, constraints, bus_cluster)
    best_cluster_bid = np.amax(cluster_bid)
    second_best_cluster_bid = np.amax(np.multiply(best_cluster_bid != cluster_bid, cluster_bid))
    if (math.floor(random.random() * rand_bid) == 0):
        best_cluster_bid = second_best_cluster_bid

    if (best_cluster_bid > 0):
        affected_list = [i for i in range(len(cluster_bid)) if ((cluster_bid[i] == best_cluster_bid) & (cluster_matrix[i, elmt] == 0))]
        n_affected_clusters = len(affected_list)

        if (n_affected_clusters != 0):
            my_idx = affected_list[math.floor(random.random() * (len(affected_list) - 1))]
            # print(np.sum(cluster_matrix, axis=1))
            new_cluster_matrix = cluster_matrix.copy()
            new_cluster_matrix[:, elmt] = 0
            new_cluster_matrix[my_idx, elmt] = 1
            [new_cluster_matrix, new_cluster_size] = trim_clusters(new_cluster_matrix)
            modified = True
    return ([modified,new_cluster_matrix,new_cluster_size])


#single thread IGTA# clustering algorithm it takes DSM,
def cluster(DSM, data, bus, constraints, p,core_parameters,objectives):
    dsm_size = DSM.shape[0]
    #pow_cc = 1; pow_bid = -2; pow_dep = 2; rand_accept = 2 * dsm_size; rand_bid = 2 * dsm_size; times = 2; stable_limit = 2
    [pow_cc,pow_bid,pow_dep,rand_accept,rand_bid,number_of_iterations,stable_limit] = core_parameters


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
            [modified,new_cluster_matrix, new_cluster_size] = move(available_elements,DSM,cluster_matrix, constraints,pow_dep, pow_bid, cluster_size,bus_cluster,rand_bid)
            if (modified):
                c = cost(DSM, new_cluster_matrix, data, pow_cc,objectives)
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

def read_matrix_with_header(file):
    with open(file,  'r') as csvfile:
        file_contents = csv.reader(csvfile,  delimiter=',',  quotechar='"')
        extracted_matrix = [c[0:len(c)] for c in file_contents]
    return (extracted_matrix)






def print_cluster(c):
    for l in c:
            print(list(l.nonzero()[0]))

def iterate_clustering(thread_index,DSM,data,constraints,bus,runs_per_thread,multi_objective,core_parameters,objectives):

    clusters = []
    if (multi_objective):
        for i in trange(runs_per_thread):
            p = i/runs_per_thread
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




def read_parameters():

    #if parameters_filename is given as argument use it otherwise use a default microcontroller file
    try:
        parameters_filename = sys.argv[1]
    except:
        parameters_filename = 'examples/dust buster parameters.csv'

    # read parameters from parameter file
    parameters = [values[0] for values in read_matrix_without_header(parameters_filename)]
    # load DSM, constraints matrix and data matrix
    DSM = np.array(read_matrix_without_header(parameters[0])).astype(int)

    # set parameters as default values
    constraints = np.ones(DSM.shape)
    constraints[18, 19] = 0
    constraints[19, 18] = 0
    data = []
    core_parameters = [1, -2, 2, 2 * DSM.shape[0], 2 * DSM.shape[0], 2 * DSM.shape[0], 2]
    number_of_threads = 1
    runs_per_thread = 100
    multi_objective = False


    #if any parameter is given in correct form use that value otherwise leave it as default
    try:
        constraints = np.array(read_matrix_without_header(parameters[1])).astype(int)
        data = np.array(read_matrix_without_header(parameters[2])).astype(float)
        core_parameters[0] = int(parameters[3])
        core_parameters[1] = int(parameters[4])
        core_parameters[2] = int(parameters[5])
        core_parameters[3] = int(parameters[6])
        core_parameters[4] = int(parameters[7])
        core_parameters[5] = int(parameters[8])
        core_parameters[6] = int(parameters[9])
        number_of_threads = int(parameters[10])
        runs_per_thread = int(parameters[11])
        if parameters[12] == 'False':
            multi_objective = False
    except:
        pass


    return [DSM,constraints,data,core_parameters,number_of_threads,runs_per_thread,multi_objective]

def convergence_population_plot(total_costs):
    points = [np.min(total_costs[0:i]) for i in range(10,len(total_costs))]
    import matplotlib.pyplot as plt
    plt.scatter(range(10,len(total_costs)), points, c='black')
    plt.show()

if __name__ == '__main__':



    [DSM, constraints, data, core_parameters, number_of_threads, runs_per_thread, multi_objective] = read_parameters()

    bus = [4,5,6,7]

    #if more than number of threads is more than 1 use a process pool otherwise use a single thread
    if (number_of_threads > 1):
        with Pool(number_of_threads) as pool:
            clustering_results = pool.map(partial(iterate_clustering, DSM=DSM, data=data, constraints=constraints, bus=bus,
                                       runs_per_thread=runs_per_thread, multi_objective=multi_objective,
                                       core_parameters=core_parameters,objectives=[1,2]), range(number_of_threads))
    else:
        clustering_results = [
            iterate_clustering(0, DSM=DSM, data=data, constraints=constraints, bus=bus,
                    runs_per_thread=runs_per_thread,multi_objective=multi_objective,
                    core_parameters=core_parameters,objectives=[1,2])]

    clustering_matrices = np.concatenate(clustering_results, axis=0);


    cluster_costs = np.array([cost(DSM, clustering_matrix, data, 1,[1,2,3]) for clustering_matrix in clustering_matrices])

    total_costs = cluster_costs[:,0]+cluster_costs[:,1]
    #find cluster with minimum total cost prints it components and costs
    min_index = np.argmin(total_costs)
    print_cluster(clustering_matrices[min_index])
    print('extra cluster cost :',cluster_costs[min_index][0],'intra cluster cost :',cluster_costs[min_index][1],'number of modules :',cluster_costs[min_index][2])

    # find pareto frontier
    pareto1 = find_pareto_points(objective1=cluster_costs[:,0],objective2=cluster_costs[:,1],number_of_pareto_points=100)

    # find pareto frontier
    pareto2 = find_pareto_points(objective1=total_costs, objective2=cluster_costs[:, 2],
                                 number_of_pareto_points=100)

    convergence_population_plot(total_costs)

    # plot pareto frontier
    if multi_objective:
        import matplotlib.pyplot as plt
        plt.scatter(cluster_costs[:,0], cluster_costs[:,1],c = 'black')
        plt.scatter(cluster_costs[pareto2, 0], cluster_costs[pareto2, 1], c='orange',marker = "^")
        plt.scatter(cluster_costs[pareto1, 0], cluster_costs[pareto1, 1], c='red',marker = "o")
        plt.scatter(cluster_costs[min_index, 0], cluster_costs[min_index, 1], c='black',s=50)
        plt.xlabel('intra cluster cost')
        plt.ylabel('extra cluster cost')
        plt.show()