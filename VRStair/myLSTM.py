#작성자 : 임수빈
#LSTM

import torch
import torch.nn as nn

dtype = torch.float

class wcLSTM(nn.Module):
    def __init__(self,num_class,input_size,hidden_size,num_layers,device):
        super(wcLSTM,self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers= num_layers, batch_first = True)
        self.W = nn.Parameter(torch.randn([hidden_size, num_class],device = self.device).type(dtype))
        self.b = nn.Parameter(torch.randn([num_class],device = self.device).type(dtype))

        
    def forward(self,x):
        x = x.transpose(0,1)
        h_0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size,requires_grad=True).to(self.device)
        c_0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size,requires_grad=True).to(self.device)

        outputs, hidden = self.lstm(x,(h_0,c_0))
        outputs = outputs[-1]
        out = torch.mm(outputs,self.W) + self.b
        return out


class PLPLSTM(nn.Module):
    def __init__(self,num_class,input_size,hidden_size,num_layers,device):
        super(PLPLSTM,self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers= num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size,num_class)

   
    def forward(self,x):
        x = x.transpose(0,1)
        h_0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size,requires_grad=True).to(self.device)
        c_0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size,requires_grad=True).to(self.device)

        outputs, hidden = self.lstm(x,(h_0,c_0))
        outputs = outputs[-1]
        out = self.fc(outputs)
        return out
