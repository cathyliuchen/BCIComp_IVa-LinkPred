import os

import numpy as np
import scipy.signal as sig
from glob import glob

# Extract subject
subjects = []
filenames = glob('epochs/p*.npy')
for file in filenames:
    subject = file.split('_')[-1][0:2]
    subjects.append(subject)
    path = 'data/adj_dict/' + subject
    if not os.path.exists(path):
        os.mkdir(path)


# Extract epochs from the numpy file for subject aa as an example
# test = np.load('epochs/preprocessed_epochs_aa.npy', allow_pickle=True)
# epochs = test.item().get('epochs')
# labels = test.item().get('labels')
# print test and epochs
# print('test:', test)
# print('epochs:', epochs)
# print('epochs shape:', np.array(epochs).shape)  # (280, 118, 350)
# print('labels shape:', np.array(labels).shape) # (280,)

 # define plv function
def cal_plv(c1, c2):
    # Hilbert transform
    c1_hill = sig.hilbert(c1)
    c2_hill = sig.hilbert(c2)
    # Phase Angle
    phase_c1 = np.unwrap(np.angle(c1_hill))
    phase_c2 = np.unwrap(np.angle(c2_hill))
    complex_phase_diff = np.exp(complex(0,1)*(phase_c1-phase_c2))
    plv = np.abs(np.sum(complex_phase_diff))/phase_c1.shape[0]
    return plv

# define adjacency matrix calculation function
def cal_adjacency_matrix(graph, node):
    adjacency_matrix = np.zeros((node, node), dtype=np.float32)
    for i in range(0, node-1):
        for j in range(0, node-1):
            ci = np.asarray(graph[i])
            cj = np.asarray(graph[j])
            #print('ci:',ci)
            #print('cj:', cj)
            adjacency_matrix[i][j] = cal_plv(ci, cj)
    return adjacency_matrix

# generate graph according to selected window
NODES = 118
TRIALS = 280
TIMESTEPS = 350
WINDOWS = 35
NO_GRAPH = int(LEN/WINDOWS)
sliced_epoch = {} # data for each sliced graph
sliced_label = {} # label for each sliced graph



for subject in subjects:
    raw_data = np.load('epochs/preprocessed_epochs_' + subject + '.npy', allow_pickle=True)
    epochs = raw_data.item().get('epochs')
    labels = raw_data.item().get('labels')
    for trial in range(0, TRIALS):
        adj_dict = {}
        for g in range(0, NO_GRAPH):
            sliced_epoch[subject + '_trail_' + str(trial+1) + '_graph_' + str(g+1)] = epochs[trial]\
                [:, g*WINDOWS:(g+1)*WINDOWS]
            # print(sub_epoch)
            graph = sliced_epoch[subject + '_trail_' + str(trial+1) + '_graph_' + str(g+1)]
            sliced_label[subject + '_trail_' + str(trial+1) + '_graph_' + str(g+1)] = labels[trial]
            adj_matrix = cal_adjacency_matrix(graph, NODES)
            temp_adj_matrix = np.array(adj_matrix)
            adj_dict[subject + '_trail_' + str(trial+1) + '_graph_' + str(g+1)] = \
                (temp_adj_matrix, sliced_label[subject + '_trail_' + str(trial+1) + '_graph_' + str(g+1)])
        print(adj_dict)


        np.save('data/adj_dict' + subject + '/' + str(trial+1) + '.npy', adj_dict)

print("Adjacent Matrix Calculation Finished.")

# Checking the adj_matrix generated of any trial. e.g. 1 of subject aa
path = 'data/adj_dict/aa/1.npy'
data = np.load(path, allow_pickle=True)
print('adj:', data)


