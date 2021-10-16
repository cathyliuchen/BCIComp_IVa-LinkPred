import numpy as np
from glob import glob
import os

# Generate folders to store dataset
subjects = ['aa', 'av', 'aw', 'al', 'ay']
for sub in subjects:
    path = 'dataset/' + sub
    if not os.path.exists(path):
        os.mkdir(path)

# reorganize the data samples for
for subject in subjects:
    trials = glob('../data/adj_dict/' + subject + '/*.npy')
    # or using sparse matrices
    # trials = glob('../data/sparse_adj_dict/' + subject + '/*.npy')
    NO_GRAPH = 10
    NODES = 118
    trial_graph = np.zeros((NO_GRAPH,NODES, NODES))
    data = []
    label = []
    for i in range(1, len(trials)+1):
        trial = np.load('../data/adj_dict/' + subject + '/' + str(i) + '.npy', allow_pickle=True).tolist()
        for item in trial.items():
            graph_id = int(item[0].split('_')[-1]) - 1
            trial_graph[graph_id] = item[1][0]
        label.append(item[1][1] - 1) # relabel 1, 2 as 0, 1 respectively.
        print("trial_graph:", trial_graph)
        data.append(trial_graph)
        print("trial_data:", data)

    # split trials into training and testing dataset：200 for training， 80 for testing.
    train_data = data[0:200]
    train_label = label[0:200]
    test_data = data[200:280]
    test_label = label[200:280]

    # save train and test dataset: data and labels.
    np.save('dataset/' + subject + '/train_data.npy', train_data)
    np.save('dataset/' + subject + '/train_label.npy', train_label)
    np.save('dataset/' + subject + '/test_data.npy', test_data)
    np.save('dataset/' + subject + '/test_label.npy', test_label)

print("Finished all subjects.")

# Check: e.g. subject aa
