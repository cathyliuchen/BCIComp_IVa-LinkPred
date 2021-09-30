import yaml
import torch as th
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from gcn_lstm_model import Classifier
from utils import DGDataset
import time

config = yaml.load(open('gnn_config.yml'), Loader=yaml.FullLoader)

node_num = config['node_num']
window_size = config['window_size']
subject = config['subject']
task = config['task']

lr = config['lr']
epochs = config['epochs']
log_interval = config['log_interval']


# load dataset
train_data_save_path = 'data/plus_dg_dataset/' + subject + '/train_data.npy'
train_label_save_path = 'data/plus_dg_dataset/' + subject + '/train_label.npy'
test_data_save_path = 'data/plus_dg_dataset/' + subject + '/test_data.npy'
test_label_save_path = 'data/plus_dg_dataset/' + subject + '/test_label.npy'

train_data = DGDataset(train_data_save_path, train_label_save_path)
test_data = DGDataset(test_data_save_path, test_label_save_path)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)

test_loader = DataLoader(
    dataset=train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)


# device configuration
th.cuda.set_device(0)


# Define model
model = Classifier(
    window_size=window_size,
    node_num=node_num,
    in_features=config['in_features'],
    out_features=config['out_features'],
    lstm_features=config['lstm_features'],
    n_class=config['num_class']
)

model = model.cuda()

# load axiliary task gcn parameters
path = '../GCN_GAN/data/result/' + subject + '/plus_rt_hand_dg_train_data_generator.pth'
save_model = th.load(path)
model_dict = model.state_dict()
state_dict = {'gcn.weights': save_model['gcn.weights']}
print(state_dict.keys())
model_dict.update(state_dict)
model.load_state_dict(model_dict)



# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(train_loader):
    model.train()
    start = time.time()
    train_loss, n_samples = 0, 0
    for batch_idx, data in enumerate(train_loader):
        train_data = data[0].cuda()
        train_label = data[1].cuda()
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        optimizer.step()
        time_iter = time.time() - start
        train_loss += loss.item() * len(output)
        n_samples += len(output)
        if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                epoch, n_samples, len(train_data),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                time_iter / (batch_idx + 1)))


def test(test_loader):
    model.eval()
    test_loss, correct, n_samples = 0, 0, 0
    for batch_idx, data in enumerate(test_loader):
        test_data = data[0].cuda()
        test_label = data[1].cuda()
        output = model(test_data)
        loss = criterion(output, test_label)
        test_loss += loss.item()
        n_samples += len(output)
        pred = output.detach().cpu().max(1, keepdim=True)[1]
        correct += pred.eq(data[1].detach().cpu().view_as(pred)).sum().item()
    test_loss /= n_samples
    acc = 100. * correct / n_samples
    print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch,
                                                                                          test_loss,
                                                                                          correct,
                                                                                          n_samples, acc))
    return acc

for epoch in range(epochs):
    train(train_loader)
acc = test(test_loader)
print(subject + ' accuracy:', acc)


print("Finished training and testing.")