import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = nn.functional.elu(self.fc1(x))
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.elu(self.fc3(x))

        return x


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

model = MLP().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10000):
    output = model(input_batch).cuda()
    loss = criterion(output, target_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


PATH = './model'
torch.save(model, PATH + "/model.pt")