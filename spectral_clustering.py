import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import csv


def seq_2_mat(seq):
    n = len(seq)
    m = max(seq)+1
    mat = np.zeros((m,n))
    mat[(seq,range(0,n))]=1
    return(mat)

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


def fix(seq):
    c = 0
    d = dict()
    n = len(seq)
    for i in range(0,n):
        if not(seq[i] in d.keys()):
            d[seq[i]] = c
            c = c + 1
        seq[i] = d[seq[i]]

def read_matrix_without_header(file):
    with open(file,  'r') as csvfile:
        file_contents = csv.reader(csvfile,  delimiter=',',  quotechar='"')
        extracted_matrix = [c[1:len(c)] for c in file_contents]
    extracted_matrix.pop(0)
    return (extracted_matrix)


DSM = np.array(read_matrix_without_header('examples/dust buster.csv')).astype(int)
sc = SpectralClustering(6, affinity='sigmoid', n_init=30,gamma =0.5,degree =3,coef0 =5)
cluster_map = sc.fit(DSM).labels_
fix(cluster_map)
c = cost(DSM, cluster_map, [], 1, [1, 2], True)
print(c)
print(0.5*c[0]+0.5*c[1])
