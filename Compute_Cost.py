import csv
import numpy as np

def fix(seq):
    c = 0
    d = dict()
    n = len(seq)
    for i in range(0,n):
        if not(seq[i] in d.keys()):
            d[seq[i]] = c
            c = c + 1
        seq[i] = d[seq[i]]


def read_list_without_header(file):
    with open(file,  'r') as csvfile:
        file_contents = csv.reader(csvfile,  delimiter=',',  quotechar='"')
        extracted_matrix = [c[0] for c in file_contents]
    return (extracted_matrix)

def read_matrix_without_header(file):
    with open(file,  'r') as csvfile:
        file_contents = csv.reader(csvfile,  delimiter=',',  quotechar='"')
        extracted_matrix = [c[1:len(c)] for c in file_contents]
    extracted_matrix.pop(0)
    return (extracted_matrix)

def seq_2_mat(seq):
    n = len(seq)
    m = max(seq)+1
    mat = np.zeros((m,n))
    mat[(seq,range(0,n))]=1
    return(mat)


def cost(DSM,  cluster_matrix,  pow_cc,objectives):

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

DSM = np.array(read_matrix_without_header("examples\car heater.csv")).astype(int)
DSM_r = np.sign(DSM+DSM.transpose())
cluster_list = np.array(read_list_without_header("examples\car heater clusters 7.csv")).astype(int)
fix(cluster_list)
cost0 = cost(DSM_r,seq_2_mat(cluster_list),1,[1,2])

print(cost0[0]+cost0[1])