import socket

import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pandas.core.indexes import interval

UnityFolder = "C:/Users/user/Desktop/Unity/VRStair/footdata/"

localIP     = "127.0.0.1"
localPort   = 8001
bufferSize  = 1024

msgFromServer       = "Hello UDP Client"
bytesToSend         = str.encode(msgFromServer)

# 데이터그램 소켓을 생성
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# 주소와 IP로 Bind
UDPServerSocket.bind((localIP, localPort))

print("UDP server up and listening")

# 들어오는 데이터그램 Listen

#plt.style.use('fivethirtyeight')

x_val = []
trackingData = []

index = count()

def DrawGraph():
    x_val.append(next(index))
    plt.plot(x_val,trackingData)


#0.01111
def animate(i):
    try:
        message = pd.read_csv(UnityFolder + "realTime.csv")
        x_val = (message["frame"])
        #trackingData.append(float(message.decode().split()[0].split(",")[1]))
        #address = bytesAddressPair[1]
        plt.cla()
        plt.plot(x_val, message["real_right"] - message["real_right"][0],label = "right(real)")
        plt.plot(x_val, message["real_left"] - message["real_left"][0],label = "left(real)")
        plt.plot(x_val, message["virtual_right"] - message["virtual_right"][0],label = "right")
        plt.plot(x_val, message["virtual_left"]- message["virtual_left"][0], label="left")

        plt.legend(loc='upper left')
    except:
        pass

    #UDPServerSocket.sendto(bytesToSend, address)

ani = FuncAnimation(plt.gcf(), animate, interval=11)


plt.tight_layout()
plt.show()

n= 0

# while(True):
#     bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
#     message = bytesAddressPair[0]
#     address = bytesAddressPair[1]
#
#     clientMsg = "Message from Client:{}".format(message)
#     clientIP  = "Client IP Address:{}".format(address)
#
#     #print(clientMsg)
#     #print(clientIP)
#     print(message.decode().split()[0].split(","))
#     trackingData.append(message.decode().split()[0].split(",")[1])
#     DrawGraph()
#     n +=1
#
#     # Sending a reply to client
#     UDPServerSocket.sendto(bytesToSend, address)



