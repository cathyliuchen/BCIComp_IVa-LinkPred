import yaml
import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from model import Generator, Discriminator
from utils import LPDataset
import os

# make result folder to store models.
subjects = ['aa', 'aw', 'av', 'al', 'ay']
for subject in subjects:
    path = 'result/' + str(subject)
    if not os.path.exists(path):
        os.mkdir(path)

# set cuda on GPU device 0.
torch.cuda.set_device(0)

# extract node number, window size, and subject
# (within subject prediction) from yaml file.
config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
node_num = config['node_num']
window_size = config['window_size']
subject = config['subject']
task = config['task']

# build and load data from dataset folder
train_data_save_path = 'dataset/' + subject + '/rt_' + task + '_train_data.npy'
train_label_save_path = 'dataset/' + subject + '/rt_' + task + '_train_label.npy'
train_data = LPDataset(train_data_save_path, train_label_save_path)
sample_data = LPDataset(train_data_save_path, train_label_save_path)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)
sample_loader = DataLoader(
    dataset=sample_data,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)

generator = Generator(
    window_size=window_size,
    node_num=node_num,
    in_features=config['in_features'],
    out_features=config['out_features'],
    lstm_features=config['lstm_features']
)

discriminator = Discriminator(
    input_size=node_num * node_num,
    hidden_size=config['disc_hidden']
)

# load model
generator = generator.cuda()
discriminator = discriminator.cuda()

# set loss
mse = nn.MSELoss(reduction='sum')
pretrain_optimizer = optim.RMSprop(generator.parameters(), lr=config['pretrain_learning_rate'])
generator_optimizer = optim.RMSprop(generator.parameters(), lr=config['g_learning_rate'])
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=config['d_learning_rate'])

# Training for generator part.
print('pretrain generator')

for epoch in range(config['pretrain_epoches']):
    for i, data in enumerate(train_loader):
        pretrain_optimizer.zero_grad()
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(),out_shot.cuda()
        predicted_shot = generator(in_shots)
        out_shot = out_shot.view(config['batch_size'], -1)
        loss = mse(predicted_shot, out_shot)
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), config['gradient_clip'])
        pretrain_optimizer.step()
        print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))

# Training for GAN part
print('train GAN')

for epoch in range(config['gan_epoches']):
    for i, (data, sample) in enumerate(zip(train_loader, sample_loader)):
        # update discriminator
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
        predicted_shot = generator(in_shots)
        _, sample = sample
        sample = sample.cuda()
        sample = sample.view(config['batch_size'], -1)
        real_logit = discriminator(sample).mean()
        fake_logit = discriminator(predicted_shot).mean()
        discriminator_loss = -real_logit + fake_logit
        discriminator_loss.backward(retain_graph=True)

        for p in discriminator.parameters():
            p.data.clamp_(-config['weight_clip'], config['weight_clip'])
        # update generator
        generator_loss = -fake_logit
        generator_loss.backward()

        discriminator_optimizer.step()
        generator_optimizer.step()
        out_shot = out_shot.view(config['batch_size'], -1)
        mse_loss = mse(predicted_shot, out_shot)
        print('[epoch %d] [step %d] [d_loss %.4f] [g_loss %.4f] [mse_loss %.4f]' % (epoch, i,
                discriminator_loss.item(), generator_loss.item(), mse_loss.item()))

# Save models with two different ways.
torch.save(generator, 'result/'+subject+'/rt_' + task + '_train_data_generator.pkl')
torch.save(generator.state_dict(), 'result/'+subject+'/rt_' + task + '_train_data_generator.pth')

print("Finished Training.")

# Check result parameters of any result e.g. aa
path = 'result/aa/rt_hand_train_data_generator.pth'
save_model = torch.load(path)
print(save_model)

