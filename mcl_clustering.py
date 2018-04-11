#!/usr/bin/env python

import sys
import numpy as np
import time
import logging
import csv
from sklearn.cluster import SpectralClustering


def normalize(A):
    column_sums = A.sum(axis=0)
    new_matrix = A / column_sums[np.newaxis, :]
    return new_matrix

def inflate(A, inflate_factor):
    return normalize(np.power(A, inflate_factor))

def expand(A, expand_factor):
    return np.linalg.matrix_power(A, expand_factor)

def add_diag(A, mult_factor):
    return A + mult_factor * np.identity(A.shape[0])

def get_clusters(A):
    clusters = []
    for i, r in enumerate((A>0).tolist()):
        if r[i]:
            clusters.append(A[i,:]>0)
    clust_map = [0 for i in range(len(A))]
    for cn , c in enumerate(clusters):
        for x in  [ i for i, x in enumerate(c) if x ]:
            clust_map[x] = cn
    return clust_map


def seq_2_mat(seq):
    n = len(seq)
    m = max(seq)+1
    mat = np.zeros((m,n))
    mat[(seq,range(0,n))]=1
    return(mat)


def print_cluster(c):
    print(c)
    for l in c:
            print(list(l.nonzero()[0]))

def stop(M, i):

    if i%5==4:
        m = np.max( M**2 - M) - np.min( M**2 - M)
        if m==0:
            logging.info("Stop at iteration %s" % i)
            return True

    return False



def seq_2_mat(seq):
    n = len(seq)
    m = max(seq)+1
    mat = np.zeros((m,n))
    mat[(seq,range(0,n))]=1
    return(mat)

def mat_2_seq(mat):
    seq = [np.nonzero(mat[:,i])[0][0] for i in range(mat.shape[1])]
    return (seq)


def cost(DSM,  clusters,  costs,  pow_cc,objectives,sequence_based):

    if sequence_based:
        cluster_matrix = seq_2_mat(clusters)
    else:
        cluster_matrix = clusters

    v = ()
    if (1 in objectives) or (3 in objectives):
        dsm_size = DSM.shape[0]
        io = np.dot(np.dot(cluster_matrix, DSM), cluster_matrix.transpose())
        ioi = io.diagonal()
        ios = np.sum(io)
        iois = np.sum(ioi)
        ioe = ios - iois
        io_extra = ioe * dsm_size
    if 1 in objectives:
        v = v + (io_extra,)

    if (2 in objectives) or (3 in objectives):
        cluster_size = np.sum(cluster_matrix, axis=1)
        cscc = np.power(cluster_size, pow_cc)
        io_intra = np.dot(ioi, cscc)
    if 2 in objectives:
        v = v + (io_intra,)


    if 3 in objectives:
        v = v + (0.5*io_intra+0.5*io_extra,)

    if 4 in objectives:
        number_of_modules = len(np.nonzero(cluster_size)[0])
        v = v + (number_of_modules,)

    return(v)


def mcl(M, expand_factor = 2, inflate_factor = 2, max_loop = 60 , mult_factor = 2):
    M = add_diag(M, mult_factor)
    M = normalize(M)

    for i in range(max_loop):
        logging.info("loop %s" % i)
        M = inflate(M, inflate_factor)
        M = expand(M, expand_factor)
        if stop(M, i): break

    clusters = get_clusters(M)
    return clusters


def fix(seq):
    c = 0
    d = dict()
    n = len(seq)
    for i in range(0,n):
        if not(seq[i] in d.keys()):
            d[seq[i]] = c
            c = c + 1
        seq[i] = d[seq[i]]

def read_matrix(csv_filename):
    import networkx as nx

    M = []
    for r in open(csv_filename):
        r = r.strip().split(",")
        M.append(list(map(lambda x: float(x.strip()), r)))

    return np.array(M)


def read_matrix_without_header(file):
    with open(file,  'r') as csvfile:
        file_contents = csv.reader(csvfile,  delimiter=',',  quotechar='"')
        extracted_matrix = [c[1:len(c)] for c in file_contents]
    extracted_matrix.pop(0)
    return (extracted_matrix)

if __name__ == '__main__':
    DSM = np.array(read_matrix_without_header('examples/dust buster.csv')).astype(int)
    clust_map = mcl(DSM)
    fix(clust_map)
    #print_cluster(seq_2_mat(clust_map))
    c = cost(DSM,clust_map,[],1,[1,2],True)
    print(c)
