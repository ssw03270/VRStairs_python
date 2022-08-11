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
# random.shuffle(tmp)

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
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = nn.functional.elu(self.fc1(x))
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.elu(self.fc3(x))

        return x

#
# model = MLP().cuda()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# for epoch in range(10000):
#     output = model(input_batch).cuda()
#     loss = criterion(output, target_batch)
#
#     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#
# PATH = './model'
# torch.save(model.state_dict(), PATH + "/model.pt")
# torch.save(model, PATH + "/model_state_dict.pt")

PATH = './model/model_state_dict.pt'
model = torch.load(PATH)
model.eval()

predict = model(test_batch)
acc = 0

real_waist = []
real_head = []
predict_waist = []
predict_head = []
for i in range(len(predict)):
    acc += abs(answer_batch[i][0] - predict[i][0]) + abs(answer_batch[i][1] - predict[i][1])
    real_waist.append(answer_batch[i][0].detach().cpu())
    real_head.append(answer_batch[i][1].detach().cpu())
    predict_waist.append(predict[i][0].detach().cpu())
    predict_head.append(predict[i][1].detach().cpu())
print(acc / len(predict))

foot_time = np.arange(len(answer_batch[:]))
plt.figure(figsize=(5, 5))
plt.title('Trajectory')
plt.xlabel('Time')
plt.ylabel('Height')
plt.plot(foot_time, real_waist)
plt.plot(foot_time, real_head)
plt.plot(foot_time, predict_waist)
plt.plot(foot_time, predict_head)
plt.legend(["real_waist", "real_head", "predict_waist", "predict_head"])
plt.show()

# Export the model
torch.onnx.export(model,                              # model being run
                  test_batch[0],                       # model dummy input (or a tuple for multiple inputs)
                  "LandingPredictor_MLP.onnx",                  # where to save the model (can be a file or file-like object)
                  export_params=True,                 # store the trained parameter weights inside the model file
                  opset_version=9,                    # the ONNX version to export the model to
                  do_constant_folding=True,           # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                'modelOutput': {0: 'batch_size'}})