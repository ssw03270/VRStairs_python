import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random
import network

cuda = torch.device('cuda')

dtype = torch.float

data = np.load('train.npz')

train_data = data['foot_train_data']
train_label = data['foot_train_label']

tmp = [[x,y] for x, y in zip(train_data, train_label)]
random.shuffle(tmp)

train_data = np.array([n[0] for n in tmp])
train_label = np.array([n[1] for n in tmp])

input_batch = torch.tensor(train_data, dtype=torch.float32, requires_grad=True, device=cuda)
target_batch = torch.tensor(train_label, dtype=torch.float32, device=cuda)

model = network.MLP().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

data_size = input_batch.shape[0]
batch_size = 32
batch_s = 0
batch_e = batch_s + batch_size
for epoch in range(50000):
    output = model(input_batch[batch_s:batch_e]).cuda()
    loss = criterion(output, target_batch[batch_s:batch_e])

    batch_s += batch_size
    batch_e += batch_size
    if batch_e > data_size:
        batch_s = 0
        batch_e = batch_s + batch_size
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


PATH = './model'
torch.save(model, PATH + "/model.pt")