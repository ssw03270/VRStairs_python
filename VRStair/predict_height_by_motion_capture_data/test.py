import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

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

train_data = data['foot_train_data'][0:1000]
train_label = data['foot_train_label'][0:1000]

input_batch = torch.tensor(train_data, dtype=torch.float32, requires_grad=True, device=cuda)
target_batch = torch.tensor(train_label, dtype=torch.float32, device=cuda)


PATH = './model/model.pt'
model = torch.load(PATH)
model.eval()

predict = model(input_batch)
acc = 0

real_waist = []
real_head = []
predict_waist = []
predict_head = []
for i in range(len(predict)):
    acc += abs(input_batch[i][0] - predict[i][0]) + abs(input_batch[i][1] - predict[i][1])
    real_waist.append(target_batch[i][0].detach().cpu())
    real_head.append(target_batch[i][1].detach().cpu())
    predict_waist.append(predict[i][0].detach().cpu())
    predict_head.append(predict[i][1].detach().cpu())
print(acc / len(predict))

foot_time = np.arange(len(target_batch[:]))
plt.figure(figsize=(5, 5))
plt.title('Trajectory')
plt.xlabel('Time')
plt.ylabel('Height')
plt.plot(foot_time, real_waist)
plt.plot(foot_time, real_head)
plt.plot(foot_time, predict_waist)
plt.plot(foot_time, predict_head)
plt.legend(["real_head", "real_waist", "predict_head", "predict_waist"])
plt.show()

# Export the model
torch.onnx.export(model,                              # model being run
                  input_batch[0],                       # model dummy input (or a tuple for multiple inputs)
                  "LandingPredictor_MLP.onnx",                  # where to save the model (can be a file or file-like object)
                  export_params=True,                 # store the trained parameter weights inside the model file
                  opset_version=9,                    # the ONNX version to export the model to
                  do_constant_folding=True,           # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                'modelOutput': {0: 'batch_size'}})