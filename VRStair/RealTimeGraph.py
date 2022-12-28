import socket

import random
from itertools import count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pandas.core.indexes import interval
from define import *
from utility import *
import seaborn as sns
import GraphMaker as g

UnityFolder = "D:/Desktop/unity/VRStair/footdata/"#"C:/Users/user/Desktop/Unity/VRStair/footdata/"


#plt.style.use('fivethirtyeight')

x_val = []
trackingData = []

index = count()

# def DrawMeanGraph(condition):
#     df = ProjectFolder + "/dataFrame/" + condition + "/"#pd.read_csv(ProjectFolder + "/dataFrame/" + condition + "/"+"trajectory.csv")
#     g.RecordedData(df).DrawPosAndVelGraph(axes)

def animate():
    try:
        message = pd.read_csv(UnityFolder + "realTime.csv")
        x_val = (message["frame"] * 0.011)
        #trackingData.append(float(message.decode().split()[0].split(",")[1]))
        #address = bytesAddressPair[1]
        plt.cla()
        #real.DrawPosGraph(additionalLabel="(real)", transX=10)
        plt.plot(x_val, message["real_right"] - message["real_right"][0],label = "right(input)")
        plt.plot(x_val, message["real_left"] - message["real_left"][0],label = "left(input)")
        plt.plot(x_val, message["virtual_head"] - message["virtual_right"][0], label="Head")
        plt.plot(x_val, message["virtual_right"] - message["virtual_right"][0],label = "RFoot")
        plt.plot(x_val, message["virtual_left"]- message["virtual_left"][0], label="LFoot")
        plt.grid(True)
    except:
        pass


def animate_vel(self):
    try:
        message = pd.read_csv(UnityFolder + "realTime.csv")
        x_val = (message["frame"] * 0.011)
        #trackingData.append(float(message.decode().split()[0].split(",")[1]))
        #address = bytesAddressPair[1]
        axes[0].cla()
        axes[1].cla()

        real.DrawPosAndVelGraph(axes,linestyle= "--", additionalLabel="(real)", transX=18)
        axes[0].plot(x_val, message["real_right"] - message["real_right"][0],label = "right(input)",color = "C3")
        axes[0].plot(x_val, message["real_left"] - message["real_left"][0],label = "left(input)",color = "C4")

        axes[0].plot(x_val, message["virtual_head"] - message["virtual_right"][0], label="Head",color = "C0")
        axes[0].plot(x_val, message["virtual_right"] - message["virtual_right"][0],label = "RFoot",color ="C1")
        axes[0].plot(x_val, message["virtual_left"]- message["virtual_left"][0], label="LFoot",color = "C2")

        r_rightVel = MakeVelData(message["real_right"],False)
        r_leftVel =  MakeVelData(message["real_left"],False)
        v_headVel = MakeVelData(message["virtual_head"],False)
        v_headVel2 = MakeVelData(message["virtual_head"] - message["real_head"] ,False)
        v_rightVel = MakeVelData(message["virtual_right"],False)
        v_leftVel = MakeVelData(message["virtual_left"],False)
        axes[1].plot(x_val,r_rightVel ,label="RFoot(input)",color = "C3")
        axes[1].plot(x_val, r_leftVel,label="LFoot(input)",color = "C4")
        axes[1].plot(x_val,v_headVel , label="head(v)",color = "C0")
        axes[1].plot(x_val, v_rightVel, label="right(v)",color = "C1")
        axes[1].plot(x_val, v_leftVel, label="left(v)",color = "C2")
        axes[1].plot(x_val, np.array(v_rightVel) - np.array(r_rightVel) , label = "right(net)",color = "C5")
        axes[1].plot(x_val, np.array(v_leftVel) - np.array(r_leftVel), label="left(net)",color = "C6")
        #axes[1].plot(x_val, v_headVel2, label="head(net)", color="C7")
    except:
        print("error")
        pass
    axes[0].set_ylim(-0.1, 2)
    axes[1].set_ylim(-1.5, 3)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')

f, axes = plt.subplots(2, 1)
condition = "stair2_75"
real = g.RecordedData(ProjectFolder + "/dataFrame/" + condition + "/",2)
#real = g.RecordedData(UnityFolder + "experiment/stair1_100_real/", 2)
#real = g.RecordedData(UnityFolder + "experiment/stair2_75_real/", 2)
#real = g.RecordedData(UnityFolder + "user3/황준태/stair2_50/3/", 2)
ani = FuncAnimation(f, animate_vel, interval=30)
#ani = FuncAnimation(plt.gcf(), animate, interval=11)
plt.tight_layout()
plt.show()





