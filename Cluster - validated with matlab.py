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


def cost(DSM,  cluster_matrix,  costs,  pow_cc):
    cluster_size = np.sum(cluster_matrix, axis=1)
    dsm_size = DSM.shape[0]
    io = np.dot(np.dot(cluster_matrix, DSM), cluster_matrix.transpose())
    ioi = io.diagonal()
    cscc = np.power(cluster_size, pow_cc)
    ios = np.sum(io)
    iois = np.sum(ioi)
    ioe = ios - iois
    io_intra = np.dot(ioi, cscc)
    io_extra = ioe*dsm_size

    number_of_modules = len(np.nonzero(cluster_size)[0])

    '''
    S1 = 0;
    for i in range(number_of_modules):
        R = np.multiply(costs[:,  0], cluster_matrix[i, :])
        non_zero_elements = np.nonzero(R)[0];
        if(len(non_zero_elements)>0):
           S1 = S1 + np.amin(R[non_zero_elements])
    MTTF = S1/number_of_modules;

    S2 = 0;
    for i in range(number_of_modules):
        R = np.multiply(costs[:,  1],  cluster_matrix[i,  :])
        non_zero_elements = np.nonzero(R)[0];
        if (len(non_zero_elements) > 0):
            S2 = S2 + np.amax(R[non_zero_elements])
    MTTR = S2 / number_of_modules;

    var_size = np.var(cluster_size[cluster_size.nonzero()[0]], ddof=1)
    '''
    return((io_intra, io_extra))




def trim_clusters(clusters):
    cluster_size = np.sum(clusters, axis=1)
    empty_clusters = np.argwhere(cluster_size == 0)
    clusters = np.delete(clusters, empty_clusters, axis=0)
    cluster_size = np.delete(cluster_size,  empty_clusters)
    return [clusters, cluster_size]

def Iplus_bid(elmt,  DSM,  Cluster_matrix,  pow_dep,  pow_bid,  cluster_size, constraints, bus_cluster):
    flag = np.zeros(Cluster_matrix.shape[0])
    flag[np.where(np.dot(Cluster_matrix, np.multiply(constraints[elmt, :], 1-bus_cluster))==0)]=1
    exclude_itself = np.identity(DSM.shape[0])
    inputs = np.dot(Cluster_matrix,  DSM[:,  elmt] - exclude_itself[:,  elmt])
    cluster_bid = np.multiply(flag, np.divide(np.power(inputs,  pow_dep), np.power(cluster_size,  pow_bid)));
    return(cluster_bid)



def read_floats(file_name):
    file = open(file_name, 'r')
    contents = file.readlines()
    values = [float(value.rstrip('\n')) for value in contents]
    return  values

rands = read_floats('rands.csv')
rand_index = 0

def rand():
    global rand_index
    global rands
    rand_index = rand_index + 1
    return(rands[rand_index-1])

def hash(x):
    return(np.dot(np.sum(x, axis=1), np.array(range(1, (len(x) + 1)))))

def cluster(DSM, costs, bus, constraints, p):
    dsm_size = DSM.shape[0]
    pow_cc = 1; pow_bid = -2; pow_dep = 2; rand_accept = 2 * dsm_size; rand_bid = 2 * dsm_size; times = 2; stable_limit = 2; max_repeat = 1000

    n_clusters = dsm_size

    bus_cluster = np.zeros(dsm_size)
    available_elements = np.ones(dsm_size)
    cluster_matrix = np.identity(dsm_size)

    for i in bus:
        cluster_matrix[i, i] = 0
        bus_cluster[i] = 1

    cluster_matrix[bus[0]]=bus_cluster
    available_elements = (np.nonzero(1-bus_cluster))[0]
    [cluster_matrix, cluster_size] = trim_clusters(cluster_matrix)
    cst = cost(DSM,  cluster_matrix,  costs,  pow_cc)
    total_cost = p*cst[0]+(1-p)*cst[1]


    stable = 0; changed = 0

    while (stable < stable_limit):
        for k in range(dsm_size*times):
            accept1 = 0
            elmt = available_elements[math.floor(rand()*(len(available_elements)-1))]
            cluster_bid = Iplus_bid(elmt,  DSM,  cluster_matrix,  pow_dep,  pow_bid,  cluster_size, constraints, bus_cluster)
            best_cluster_bid = np.amax(cluster_bid)
            second_best_cluster_bid = np.amax(np.multiply(best_cluster_bid != cluster_bid, cluster_bid))
            if (math.floor(rand()*rand_bid) == 0):
                best_cluster_bid = second_best_cluster_bid

            if (best_cluster_bid>0):
                affected_list = [i for i in range(len(cluster_bid)) if ((cluster_bid[i]==best_cluster_bid)&(cluster_matrix[i, elmt]==0))]
                n_affected_clusters = len(affected_list)

                if (n_affected_clusters == 0):
                    continue

                my_idx = affected_list[math.floor(rand()*(len(affected_list)-1))]
               # print(np.sum(cluster_matrix, axis=1))
                new_cluster_matrix = cluster_matrix.copy()
                new_cluster_matrix[:,  elmt] = 0
                new_cluster_matrix[my_idx,  elmt] = 1
                [new_cluster_matrix, new_cluster_size] = trim_clusters(new_cluster_matrix)
                c = cost(DSM, new_cluster_matrix, costs, pow_cc)
        #        print([c[0],c[1],rand_index,hash(new_cluster_matrix)])
        #        if ((c[0] == 116) & (c[1] == 4700)):
        #            print('here')

                new_total_cost = p * c[0] + (1 - p) * c[1]
                if ((new_total_cost <= total_cost) | (math.floor(rand_accept * rand()) == 0)):
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
    print(total_cost)
    print(rand_index)
    return [cluster_matrix,total_cost]




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


def run(DSM,costs,constraints,bus):
    p = 0.5;

    clusters = []
    for i in range(1000):
        print(i)
        clusters.append(cluster(DSM, costs, bus, constraints, p))

    costs = [c[1] for c in clusters]
    min_index = costs.index(min(costs))
    print_cluster(clusters[min_index][0])
    print(clusters[min_index][1])





DSM = np.array(read_matrix_without_header('examples/contrast injector.csv')).astype(int)
cluster_matrix = np.array(read_matrix_with_header('examples/cluster.csv')).astype(int)
costs = np.array(read_matrix_without_header('examples/UAV_costs.csv')).astype(float)
constraints = np.array(read_matrix_without_header('examples/UAV_constraints.csv')).astype(int)
constraints = np.zeros((len(DSM),len(DSM)))
constraints[8,9]=1;constraints[9,8]=1


#print(costs)
bus = [3, 4]
run(DSM,costs,constraints,bus)
