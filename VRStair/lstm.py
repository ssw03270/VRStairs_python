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

# train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)

class LSTM(nn.Module):
    def __init__(self, num_class, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.W = nn.Parameter(torch.randn([hidden_size, num_class], device=cuda).type(dtype))
        self.b = nn.Parameter(torch.randn([num_class], device=cuda).type(dtype))

    def forward(self, x):
        x = x.transpose(0, 1)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True, device=cuda)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True, device=cuda)

        outputs, hidden = self.lstm(x, (h_0, c_0))
        outputs = outputs[-1]
        out = torch.mm(outputs, self.W) + self.b
        return out

num = int(len(train_data) * 0.9)

input_batch = torch.tensor(train_data[0:num], dtype=torch.float32, requires_grad=True, device=cuda)
target_batch = torch.tensor(train_label[0:num], dtype=torch.float32, device=cuda)

test_batch = torch.tensor(train_data[num:len(train_data)], dtype=torch.float32, requires_grad=True, device=cuda)
answer_batch = torch.tensor(train_label[num:len(train_data)], dtype=torch.float32, device=cuda)

print(input_batch, target_batch)
print("test data:", len(test_batch), "\n", answer_batch)

model = LSTM(60, 6, 8, 3).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    # if epoch == 200:
    #     optimizer.param_groups[0]['lr'] = 0.001
    # if epoch == 400:
    #     optimizer.param_groups[0]['lr'] = 0.0001

    output = model(input_batch).cuda()
    loss = criterion(output, target_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

PATH = './model'
torch.save(model.state_dict(), PATH + "/model.pt")
torch.save(model, PATH + "/model_state_dict.pt")

# PATH = 'model/model_state_dict.pt'
# model = torch.load(PATH)
# model.eval()

predict = model(test_batch)
acc = 0

for i in range(len(predict)):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(predict[i][0::6].detach().cpu().numpy(), predict[i][1::6].detach().cpu().numpy(), predict[i][2::6].detach().cpu().numpy(), 'r')
    ax.plot(answer_batch[i][0::6].cpu().numpy(), answer_batch[i][1::6].cpu().numpy(), answer_batch[i][2::6].cpu().numpy(), 'g')
    # ax.plot(test_batch[i][0::12].detach().cpu().numpy(), test_batch[i][1::12].detach().cpu().numpy(), test_batch[i][2::12].detach().cpu().numpy(), 'b')
    ax.set_xlabel('axis_x')
    ax.set_ylabel('axis_y')
    ax.set_zlabel('axis_z')
    plt.show()


# dummy_in = torch.randn(1, 10, 7, device=cuda)
# print(dummy_in.shape)
# torch.onnx.export(model,                              # model being run
#                   dummy_in,                        # model dummy input (or a tuple for multiple inputs)
#                   "lstm_classification.onnx",         # where to save the model (can be a file or file-like object)
#                   export_params=True,                 # store the trained parameter weights inside the model file
#                   opset_version=9,                    # the ONNX version to export the model to
#                   do_constant_folding=True,           # whether to execute constant folding for optimization
#                   input_names = ['x'],                # the model's input names
#                   output_names = ['y']                # the model's output names
#                   )