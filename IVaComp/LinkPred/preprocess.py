import numpy as np
from glob import glob
import os
subjects = ['aa', 'aw', 'av', 'al', 'ay']

# load data
rt_hand_data = []
rt_hand_label = []
rt_foot_data = []
rt_foot_label = []

for subject in subjects:
    path = 'dataset/' + str(subject)
    if not os.path.exists(path):
        os.mkdir(path)

    trialnames = glob('../data/saprse_adj_dict/' + subject + '/*.npy')
    for trial in trialnames:
        adj_dict = np.load(trial,allow_pickle=True)

        trial_data = []
        trial_label = []
        for item in adj_dict.tolist().items():
            trial_data.append(item[1][0])
            trial_label.append(item[1][1])

        sample_1 = np.asarray(trial_data[0:4]).astype('float32')  # (5, 118, 118)
        sample_2 = np.asarray(trial_data[1:5]).astype('float32')
        sample_3 = np.asarray(trial_data[2:6]).astype('float32')
        sample_4 = np.asarray(trial_data[3:7]).astype('float32')
        sample_5 = np.asarray(trial_data[4:8]).astype('float32')
        sample_6 = np.asarray(trial_data[5:9]).astype('float32')

        label_1 = np.asarray(trial_data[4]).astype('float32')
        label_2 = np.asarray(trial_data[5]).astype('float32')  # (118, 118)
        label_3 = np.asarray(trial_data[6]).astype('float32')
        label_4 = np.asarray(trial_data[7]).astype('float32')
        label_5 = np.asarray(trial_data[8]).astype('float32')
        label_6 = np.asarray(trial_data[9]).astype('float32')

        if trial_label[0] == 1:
            rt_hand_data.extend([sample_1, sample_2,sample_3, sample_4, sample_5, sample_6])
            rt_hand_label.extend([label_1, label_2,  label_3, label_4, label_5, sample_6])
        elif trial_label[0] == 2:
            rt_foot_data.extend([sample_1, sample_2, sample_3, sample_4, sample_5, sample_6])
            rt_foot_label.extend([label_1, label_2, label_3, label_4, label_5, sample_6])

    rt_hand_train_data = rt_hand_data[0:600]
    rt_hand_train_label = rt_hand_label[0:600]
    rt_hand_test_data = rt_hand_data[600:700]
    rt_hand_test_label = rt_hand_label[600:700]

    rt_foot_train_data = rt_foot_data[0:600]
    rt_foot_train_label = rt_foot_label[0:600]
    rt_foot_test_data = rt_foot_data[600:700]
    rt_foot_test_label = rt_foot_label[600:700]


    # save data
    np.save('dataset/' + subject +'/rt_hand_train_data.npy', rt_hand_train_data)
    np.save('dataset/' + subject +'/rt_hand_train_label.npy', rt_hand_train_label)
    np.save('dataset/' + subject +'/rt_hand_test_data.npy', rt_hand_test_data)
    np.save('dataset/' + subject +'/rt_hand_test_label.npy', rt_hand_test_label)

    np.save('dataset/' + subject +'/rt_foot_train_data.npy', rt_foot_train_data)
    np.save('dataset/' + subject +'/rt_foot_train_label.npy', rt_foot_train_label)
    np.save('dataset/' + subject +'/rt_foot_test_data.npy', rt_foot_test_data)
    np.save('dataset/' + subject +'/rt_foot_test_label.npy', rt_foot_test_label)


print("Finished with all subjects.")
