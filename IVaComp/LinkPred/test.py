import yaml
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from utils import LPDataset
from utils import MSE, EdgeWiseKL, MissRate

# load config parameters.
config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
node_num = config['node_num']
window_size = config['window_size']
subject = config['subject']
task = config['task']

# load test data and model.
generator = torch.load('result/' + subject + '/rt_' + task + '_train_data_generator.pkl').cuda()
test_data_save_path = 'dataset/' + subject + '/rt_' + task + '_test_data.npy'
test_label_save_path = 'dataset/' + subject + '/rt_' + task +'_test_label.npy'

# build test dataset.
test_data = LPDataset(test_data_save_path, test_label_save_path)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True
)

# initialize the valuation metrics.
total_samples = 0
total_mse = 0
total_kl = 0
total_missrate = 0

for i, data in enumerate(test_loader):
    in_shots, out_shot = data
    in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
    predicted_shot = generator(in_shots)
    predicted_shot = predicted_shot.view(-1, config['node_num'], config['node_num'])
    predicted_shot = (predicted_shot + predicted_shot.transpose(1, 2)) / 2
    for j in range(config['node_num']):
        predicted_shot[:, j, j] = 0
    mask = predicted_shot >= config['epsilon']
    predicted_shot = predicted_shot * mask.float()
    batch_size = in_shots.size(0)
    total_samples += batch_size
    total_mse += batch_size * MSE(predicted_shot, out_shot)
    total_kl += batch_size * EdgeWiseKL(predicted_shot, out_shot)
    total_missrate += batch_size * MissRate(predicted_shot, out_shot)

MSE = 'MSE: %.4f' % (total_mse / total_samples)
edge_wise_KL = 'edge wise KL: %.4f' % (total_kl / total_samples)
miss_rate = 'miss rate: %.4f' % (total_missrate / total_samples)
print(MSE)
print(edge_wise_KL)
print(miss_rate)

# save result
test_result = [MSE, edge_wise_KL, miss_rate]
np.save('result/' + subject + '/' + task +'_test.npy', test_result)