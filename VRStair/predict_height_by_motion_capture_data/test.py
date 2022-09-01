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

train_data = data['foot_train_data'][100000:102000]
train_label = data['foot_train_label'][100000:102000]

input_batch = torch.tensor(train_data, dtype=torch.float32, requires_grad=True, device=cuda)
target_batch = torch.tensor(train_label, dtype=torch.float32, device=cuda)


PATH = './model/model.pt'
model = torch.load(PATH)
model.eval()

predict = model(input_batch)
acc = 0

real_head = []
predict_head = []

for i in range(len(predict)):
    acc += abs(target_batch[i] - predict[i])
    real_head.append(target_batch[i].detach().cpu())
    predict_head.append(predict[i].detach().cpu())

print(acc / len(predict))

foot_time = np.arange(len(target_batch))
plt.figure(figsize=(5, 5))
plt.title('Trajectory')
plt.xlabel('Time')
plt.ylabel('Height')
plt.plot(foot_time, real_head)
plt.plot(foot_time, predict_head)
plt.legend(["real_head", "predict_head"])
plt.show()

# Export the model
torch.onnx.export(model,                              # model being run
                  input_batch,                       # model dummy input (or a tuple for multiple inputs)
                  "height_predict_model.onnx",                  # where to save the model (can be a file or file-like object)
                  export_params=True,                 # store the trained parameter weights inside the model file
                  opset_version=9,                    # the ONNX version to export the model to
                  do_constant_folding=True,           # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                'modelOutput': {0: 'batch_size'}})