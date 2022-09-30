
import GraphMaker as g
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

folder = "C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
Rvelocity =  g.loadData(folder+"Rvelocity.txt")
Lvelocity =  g.loadData(folder+"Lvelocity.txt")

f, axes = plt.subplots(2, 1)
# #virtual = g.RecordedData(folder+"recodingTest/test/",1).DrawPosAndVelGraph(axes)
# # virtual = g.RecordedData(folder+"recodingTest/test2/",1).DrawPosAndVelGraph(axes, color="C1")
real = g.RecordedData(folder+"real/2/",2).DrawPosAndVelGraph(axes,startIndex=30, color="C1",additionalLabel=" (real)")
#real = g.RecordedData(folder+"real/3/",2).DrawPosAndVelGraph(axes,startIndex=30, color="C1",additionalLabel=" (real)")
#my = g.RecordedData(folder+"my/1/",2).DrawPosAndVelGraph(axes,startIndex=30, color="r", additionalLabel= " (my)")
my = g.RecordedData(folder+"my/2/",2).DrawPosAndVelGraph(axes,startIndex=30, color="r", additionalLabel= " (my)")
#my = g.RecordedData(folder+"my/3/",2).DrawPosAndVelGraph(axes,startIndex=30, color="r", additionalLabel= " (my)")
xAxis = np.array(list(range(0, len(Rvelocity[1])))) * g.fixedDeltaTime
axes[1].plot(xAxis,Rvelocity[1])
axes[1].plot(xAxis,Lvelocity[1])
#virtual = g.RecordedData(folder+"recodingTest/sp4/",1).DrawPosAndVelGraph(axes)
#folder2 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
#real = g.RecordedData(folder2,2)
