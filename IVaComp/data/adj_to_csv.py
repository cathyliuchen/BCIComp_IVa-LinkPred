import pandas as pd
import numpy as np
from glob import glob
import os
import shutil
import csv

# Generate folders to store csv files
subjects = ['aa', 'aw', 'al', 'av', 'ay']

for sub in subjects:
    csv_path = 'adj_to_csv/' + sub + '/edges'
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

# Generate dg_adj_dict for 280 trials for 5 subjects.
for sub in subjects:
    trials = glob('sparse_adj_dict/' + sub + '/*.npy')
    # generate trial folders to store graphs(.csv).
    for trial in range(1, len(trials)+1):
        trial_path = '../data/adj_to_csv/' + sub + '/trial_' + str(trial)
        if not os.path.exists(trial_path):
            os.mkdir(trial_path)

    # move graphs(.csv) to corresponding trial folder
    graphs = glob('adj_to_csv/' + sub + '/edges/*.csv')
    for graph in graphs:
        trial_id = graph.split('/')[-1].split('_')[1]
        graph_id = graph.split('/')[-1].split('_')[3][0]
        trial_path = 'adj_to_csv' + sub + '/edges/trial_' + str(trial_id)
        shutil.move(graph, trial_path)

    NODES = 118
    for trial in range(1, len(trials)+1):
        trial = np.load('adj_dict/'+ sub + '/' + str(trial) + '.npy', allow_pickle=True).tolist()
        for item in trial.items():
            gkey = item[0][3:]
            graph = np.asarray(item[1][0])
            # glabel = np.asarray(item[1][1])
            csv_path = 'adj_to_csv/' + sub + '/edges' + gkey +'.csv'
            data = []
            for i in range(0, NODES):
                for j in range(0, NODES):
                    data.append([i, j, graph[i][j]])

            df = pd.DataFrame(data, columns=['src', 'dst', 'weight'])
            df.to_csv(csv_path)
        print('Finished for ' + sub + '' + str(trial))
    print('Finished all trials for ' + sub)
print('Finished for all subject')


# check csv file of any graph. e.g. subject aa, trial 1, graph 1
file = open("adj_to_csv/aa/edges/trail_1/trial_1_graph_1.csv")
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows=[]
for row in csvreader:
    rows.append(row)
print(rows)
file.close()

























