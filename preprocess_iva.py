import numpy as np
from glob import glob

# load data
sub = 'av'
rt_hand_data = []
rt_foot_data = []

filenames = glob('data/adj_dict/'+ sub +'/*.npy')
for file in filenames:
    adj_dict = np.load(file,allow_pickle=True)

    for item in adj_dict.tolist().items():
        trial_data = item[1][0]
        trial_label = item[1][1]

        sample_1 = trial_data[0:5]
        sample_2 = trial_data[1:6]
        sample_3 = trial_data[2:7]
        sample_4 = trial_data[3:8]
        sample_5 = trial_data[4:9]
        label_1 = trial_data[5]
        label_2 = trial_data[6]
        label_3 = trial_data[7]
        label_4 = trial_data[8]
        label_5 = trial_data[10]

        if trial_label == 1:
            rt_hand_data.extend([(sample_1, label_1), (sample_2, label_2),
                                 (sample_3, label_3), (sample_4, label_4),
                                 (sample_5, label_5)])
        elif trial_label == 2:
            rt_foot_data.extend([(sample_1, label_1), (sample_2, label_2),
                                 (sample_3, label_3), (sample_4, label_4),
                                 (sample_5, label_5)])

rt_hand_train_data = rt_hand_data[0:600]
rt_hand_test_data = rt_hand_data[600:700]
rt_foot_train_data = rt_foot_data[0:600]
rt_foot_test_data = rt_foot_data[600:700]
# save data

np.save('data/dataset/rt_hand_dataset/rt_hand_train_data_' + sub + '.npy', rt_hand_train_data)
np.save('data/dataset/rt_hand_dataset/rt_hand_test_data_' + sub + '.npy', rt_hand_test_data)
np.save('data/dataset/rt_foot_dataset/rt_foot_train_data_' + sub + '.npy', rt_foot_train_data)
np.save('data/dataset/rt_foot_dataset/rt_foot_test_data_' + sub + '.npy', rt_foot_test_data)

print("Finished with subject " + sub)
