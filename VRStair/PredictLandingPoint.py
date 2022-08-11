#작성자 : 임수빈
#착지점 예측 모델
#

import FootDataProcessor2 as fp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import myLSTM as mL
import math
import torch.onnx

'''
0 :1step 앞으로
1 :연속적으로 앞으로
2 : 옆으로 1step
3 : 뒤로 1
4 : 뒤로 2

5 : 계단올라가는 듯이 1
6 : 계단 올라가듯이 2
7 : 옆으로 계단올라가듯이 1
8 : 뒤로 올라가듯이 1 
9 : 뒤로 올라가듯이 2

'''
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class myCNN(nn.Module):
    def __init__(self,featureNum,seq,device):
        super().__init__()
        self.featureNum = featureNum
        self.seq = seq
        self.conv1 = nn.Conv1d(in_channels = 1,out_channels = 1,padding=36, kernel_size = seq, stride = 1, dilation = featureNum)
        self.inputNum = seq * featureNum
        self.fc1 = nn.Linear( self.inputNum, self.inputNum * 2, bias = True , device = device)
        self.fc2 = nn.Linear( self.inputNum * 2, self.inputNum, bias = True , device = device)
        self.fc3 = nn.Linear( self.inputNum, 2, bias = True , device = device)
        #self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = torch.tanh(x1)
        x1 = x1.reshape(-1, self.inputNum)
        out = torch.tanh(self.fc1(x1))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        #print(out.squeeze().shape)
        return out
                
class myMLP(nn.Module):
    def __init__(self,inputNum,device):
        super().__init__()
        self.fc1 = nn.Linear(inputNum,inputNum*2,bias = True,device = device)
        self.fc2 = nn.Linear(inputNum*2,inputNum,bias = True,device = device)
        self.fc3 = nn.Linear(inputNum,2,bias = True,device = device)
        #self.fc4 = nn.Linear(inputNum-5,3,bias = True,device = device)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        #x = x.transpose(0,1)
        x = torch.tanh(self.fc1(x))#torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
#        x = torch.tanh(self.fc4(x))
        return x

#print(Device)
input_batch, target_batch,test_batch,answer_batch,pos_data = fp.MakeData4(10)
#    np.savez("footTrain",input = trainInput,target = trainTarget)
#    np.savez("footTest", input = testInput, target = testTarget)
'''
trainLoad = np.load("footTrain.npz")
testLoad = np.load("footTest.npz")
input_batch = trainLoad['input']
target_batch = trainLoad['target']

test_batch = testLoad['input']
answer_batch = testLoad['target']

input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True).to(Device)
target_batch = torch.tensor(target_batch, dtype=torch.float32).to(Device)

test_batch = torch.tensor(test_batch, dtype=torch.float32, requires_grad=False).to(Device)
answer_batch = torch.tensor(answer_batch, dtype=torch.float32).to(Device)
'''

#print(input_batch,target_batch)
#for i in test_batch:
#    print(i);
print("test data:", len(test_batch), "\n",answer_batch)
print(input_batch.size())
walkingTypes = ['']

#model = mL.PLPLSTM(2,8,6,1,Device).to(Device)
model = myMLP(27,Device).to(Device)
#model = myCNN(8,10,Device).to(Device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()

'''
for epoch in range(2000):
    output = model(input_batch).to(Device)
    loss = criterion(output,target_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
#for cnn training
#input_batch = input_batch.reshape(1,1,-1)
#print(input_batch.size())

for epoch in range(1800):
    output = model(input_batch).to(Device)
    loss = criterion(output,target_batch)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


'''
    # Export the model
torch.onnx.export(model,                              # model being run
                  test_batch[0].reshape(1,1,-1),                       # model dummy input (or a tuple for multiple inputs)
                  "LandingPredictor_CNN.onnx",                  # where to save the model (can be a file or file-like object)
                  export_params=True,                 # store the trained parameter weights inside the model file
                  opset_version=9,                    # the ONNX version to export the model to
                  do_constant_folding=True,           # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                'modelOutput': {0: 'batch_size'}})

model.eval()

'''
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
#'''
PATH = './model'
torch.save(model.state_dict(), PATH + "/model.pt")
torch.save(model, PATH + "/model_state_dict.pt")

#predict = model(test_batch.reshape(1,1,-1)).data
predict = model(test_batch).data
acc = 0
ed = 0

totalErrorList = []
curAnswer = answer_batch[0]
stepSize = 0
curErrorList = []
curDistance = math.sqrt(curAnswer[0]**2 + curAnswer[1]**2)
#print(len(predict))
for i in range(len(predict)):
    #print(predict[i],answer_batch[i])
    #print(i,"/",len(predict))
    sq = (predict[i][0]-answer_batch[i][0])**2 + (predict[i][1] - answer_batch[i][1])**2
    ssq = math.sqrt(sq)
    ed += ssq
    if (answer_batch[i][0] != curAnswer[0]) or (answer_batch[i][1] != curAnswer[1]):
        #print(stepSize)
        totalErrorList.append((curErrorList.copy(),curDistance))
        curErrorList = []
        curAnswer = answer_batch[i]
        curDistance = math.sqrt(curAnswer[0] ** 2 + curAnswer[1] ** 2)
        stepSize += 1
    curErrorList.append(ssq)

sampleSize = 100
errorSampleList = [0] * sampleSize
meanTotalMoving = 0
meanMSE = 0

for data in totalErrorList:
    for i in range(0,sampleSize):
        curIndex= round((len(data[0])-1) * (i+1)/sampleSize)
        errorSampleList[i] += (data[0][curIndex] / data[1])
        #print(data.MSE, data.totalMovingDistance)

n = 0
x_values = []
y_values = []
dataSize = len(totalErrorList)
for e in errorSampleList:
    x_values.append(n)
    y_values.append(e/dataSize)
    n += 1

plt.plot(x_values,y_values)
plt.show()


print("MSE: ",ed/len(predict) , "\nlast error rate:", errorSampleList[sampleSize-1]/dataSize, "\nfirst error rate : ", errorSampleList[0]/dataSize)



'''


i = 0
for data,num in pos_data:
    xPos = []
    zPos = []
    inputX = []
    inputZ = []
    k = 0
    for d in data:   
        xPos.append(d.x)
        zPos.append(d.z)
        if k <= num: 
            inputX.append(d.x)
            inputZ.append(d.z)
        k += 1
        
    #plt.plot(xPos.copy(),yPos.copy(),'r')
    #plt.plot(predict[i][0].item(),predict[i][1].item(),'bo')
    #plt.plot(answer_batch[i][0].item(), answer_batch[i][1].item(),'go')
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.plot(xPos.copy(),zPos.copy(),'r')
    plt.plot(inputX,inputZ,'g')
    plt.plot(predict[i][0].item(),predict[i][1].item(),'bo')
    plt.plot(answer_batch[i][0].item(), answer_batch[i][1].item(),'go')
    #plt.show()
    plt.savefig('./img3/'+'data'+str(i)+'.png')
    plt.cla()
    #plt.close(fig)
    i += 1
    '''

'''
for i in range(len(predict)):
    #print(predict[i],answer_batch[i])
    sq = (predict[i][0]-answer_batch[i][0])**2 + (predict[i][1]- answer_batch[i][1])**2 + (predict[i][2]-answer_batch[i][2])**2
    ed += math.sqrt(sq)
print(ed/len(predict)) 



#시각화
i = 0
for data,num in pos_data:
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    xPos = []
    yPos = []
    zPos = []
    inputX = []
    inputY = []
    inputZ = []
    k = 0
    for d in data:   
        xPos.append(d.x)
        yPos.append(d.y)
        zPos.append(d.z)
        if k <= num: 
            inputX.append(d.x)
            inputY.append(d.y)
            inputZ.append(d.z)
        k += 1
        
    #plt.plot(xPos.copy(),yPos.copy(),'r')
    #plt.plot(predict[i][0].item(),predict[i][1].item(),'bo')
    #plt.plot(answer_batch[i][0].item(), answer_batch[i][1].item(),'go')
    #plt.xlabel('x')
    #plt.ylabel('y')
    ax.set_xlabel('axis_x')
    ax.set_ylabel('axis_y')
    ax.set_zlabel('axis_z')
    ax.plot(xPos.copy(),yPos.copy(),zPos.copy(),'r')
    ax.plot(inputX,inputY,inputZ,'g')
    ax.plot(predict[i][0].item(),predict[i][1].item(),predict[i][2].item(),'bo')
    ax.plot(answer_batch[i][0].item(), answer_batch[i][1].item(),answer_batch[i][2].item(),'go')
    plt.show()
    i += 1
    '''

