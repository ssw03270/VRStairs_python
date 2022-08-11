import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

cuda = torch.device('cuda')

dtype = torch.float

data = np.load('train.npz')

train_data = data['foot_train_data']
train_label = data['foot_train_label']

# train_data /= np.linalg.norm(train_data)

tmp = [[x,y] for x, y in zip(train_data, train_label)]
random.shuffle(tmp)

train_data = np.array([n[0] for n in tmp])
train_label = np.array([n[1] for n in tmp])

num = int(len(train_data) * 0.9)

input_batch = torch.tensor(train_data[0:num], dtype=torch.float32, requires_grad=True, device=cuda)
target_batch = torch.tensor(train_label[0:num], dtype=torch.float32, device=cuda)

test_batch = torch.tensor(train_data[num:len(train_data)], dtype=torch.float32, requires_grad=True, device=cuda)
answer_batch = torch.tensor(train_label[num:len(train_data)], dtype=torch.float32, device=cuda)

print(input_batch, target_batch)
print("test data:", len(test_batch), "\n", answer_batch)

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(120, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 120)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = nn.functional.elu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.elu(self.fc2(x))
        x = self.dropout(x)
        x = nn.functional.elu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x


model = MLP().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(3000):
    output = model(input_batch).cuda()
    loss = criterion(output, target_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


PATH = './model'
torch.save(model.state_dict(), PATH + "/model.pt")
torch.save(model, PATH + "/model_state_dict.pt")

# PATH = './model/model_state_dict.pt'
# model = torch.load(PATH)
# model.eval()

predict = model(test_batch)
acc = 0

for i in range(len(predict)):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predict[i][0::6].detach().cpu().numpy(), predict[i][1::6].detach().cpu().numpy(), predict[i][2::6].detach().cpu().numpy(), 'r')
    ax.plot(answer_batch[i][0::6].cpu().numpy(), answer_batch[i][1::6].cpu().numpy(), answer_batch[i][2::6].cpu().numpy(), 'g')
    ax.plot(test_batch[i][0::6].detach().cpu().numpy(), test_batch[i][1::6].detach().cpu().numpy(), test_batch[i][2::6].detach().cpu().numpy(), 'b')
    ax.set_xlabel('axis_x')
    ax.set_ylabel('axis_y')
    ax.set_zlabel('axis_z')
    plt.show()