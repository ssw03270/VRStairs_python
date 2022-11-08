
import GraphMaker as g
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

folder = "C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
Rvelocity =  g.loadData(folder+"Rvelocity.txt")
Lvelocity =  g.loadData(folder+"Lvelocity.txt")

start = 5
f, axes = plt.subplots(2, 1)
#virtual = g.RecordedData(folder+"for_test/test/",1).DrawPosAndVelGraph(axes, color="C1")
#virtual = g.RecordedData(folder+"for_test/test/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(me)")
#virtual = g.RecordedData(folder+"for_test/10-26/24/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(me)")
# virtual = g.RecordedData(folder+"recodingTest/test3/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(1)")
# virtual = g.RecordedData(folder+"recodingTest/test7/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(2)")
#virtual = g.RecordedData(folder+"recodingTest/test4/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(3)")
virtual = g.RecordedData(folder+"recodingTest/test8/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(boundary gaussian)")
#virtual = g.RecordedData(folder+"for_test/stair1_75/1/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(real Time record)")
#virtual = g.RecordedData(folder+"recodingTest/test6/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(6)")
#g.RecordedData(folder+"for_test/waist50/2/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (waist based)",addtionalHeight=0.3,transX=-0) #50
#g.RecordedData(folder+"for_test/waist100/2/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (waist based)",addtionalHeight=0.3,transX=-220) # 100
#g.RecordedData(folder+"for_test/10-21/3/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (waist based)",addtionalHeight=0.3,transX=-80) #75
#real = g.RecordedData(folder+"user3/임수빈/stair1_100/1/",2).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (real stair)",addtionalHeight=0.27,transX=16)
#real = g.RecordedData(folder+"user3/서승원/stair2_75/4/",2).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (real stair)",addtionalHeight=-0.5,transX=20)
real = g.RecordedData(folder+"real/4/",2).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (real stair)",addtionalHeight=-0.5,transX=0)
#axes[1].axhline(0.5,0,1,linestyle='--')
axes[0].set_title("position")
axes[1].set_title("vertical velocity")

plt.rc('font', size=10)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
#plt.rc('legend', fontsize=20)  # 범례 폰트 크기
#plt.rc('figure', titlesize=50) # figure title 폰트 크기
plt.ylim(-2,4)
axes[0].set_ylabel('height(m)')
axes[1].set_xlabel('time(s)')
axes[1].set_ylabel('velocity(m/s)')
#real = g.RecordedData(folder+"real/3/",2).DrawPosAndVelGraph(axes,startIndex=30, color="C1",additionalLabel=" (real)")
#my = g.RecordedData(folder+"my/1/",2).DrawPosAndVelGraph(axes,startIndex=30, color="r", additionalLabel= " (my)")
#my = g.RecordedData(folder+"my/3/",2).DrawPosAndVelGraph(axes,startIndex=30, color="r", additionalLabel= " (my)")
#my = g.RecordedData(folder+"my/4/",2).DrawPosAndVelGraph(axes,startIndex=start, color="r", additionalLabel= " (my)")
# xAxis = np.array(list(range(start, len(Rvelocity[1])))) * g.fixedDeltaTime
# axes[1].plot(xAxis,Rvelocity[1][start:],color="pink")
# axes[1].plot(xAxis,Lvelocity[1][start:],color="pink")
plt.show()
#virtual = g.RecordedData(folder+"recodingTest/sp4/",1).DrawPosAndVelGraph(axes)
#folder2 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
#real = g.RecordedData(folder2,2)
