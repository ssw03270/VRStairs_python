
import GraphMaker as g
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns



ProjectFolder = os.getcwd()  #"C:/Users/user/Desktop/Unity/VRStair/footdata/"#"C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
ProjectFolder = ProjectFolder.replace("\\","/",10)
UnityFolder = "D:/Desktop/unity/VRStair/footdata/" #"C:/Users/user/Desktop/Unity/VRStair/footdata/"
folder = UnityFolder

# Rvelocity =  g.loadData(folder+"Rvelocity.txt")
# Lvelocity =  g.loadData(folder+"Lvelocity.txt")

start = 5
f, axes = plt.subplots(2, 1)
#analyze(folder,axes)


def ReadDataFrame(condition,type):
    df = pd.read_csv(ProjectFolder + "/dataFrame/" + condition+"/"+type+ ".csv")
    sns.lineplot(x="time", y="velY", data=df,ax=axes[1],label = condition)

#d1 = g.DrawAVGHeadGraph(axes,"stair2_75")
# for i in range(0,10):
#     g.ReadAndDrawGraph(folder+"user3/황주영/stair2_75/"+str(i)+"/LfootRotationData.txt",folder+"user3/황주영/stair2_75/"+str(i)+"/RfootRotationData.txt")
#     g.ReadAndDrawGraph(folder + "user3/서승원/stair2_75/" + str(i) + "/LfootRotationData.txt",
#                     folder + "user3/서승원/stair2_75/" + str(i) + "/RfootRotationData.txt",rcolor="gray",lcolor="C2")
    #plt.show()
#virtual = g.RecordedData(folder+"recodingTest/vrstair/stair2_50/",1).DrawHeadGraph(axes,startIndex=start,avgInfo=d1,additionalLabel="(vrstair)")
# virtual = g.RecordedData(folder + "recodingTest/our/stair2_100/1/", 1).DrawHeadGraph(axes, startIndex=start,
#                                                                                                  avgInfo=d1,
#                                                                                                  additionalLabel="(boundary gaussian)")
# for i in range(4):
#     virtual = g.RecordedData(folder+"recodingTest/our/stair1_50/"+str(i)+"/",1).DrawHeadGraph(axes,startIndex=start,avgInfo=d1,additionalLabel="(boundary gaussian)")


#virtual = g.RecordedData(UnityFolder+"ex3_test/stiar2_50_ours/0/",1).DrawHeadGraph(axes,startIndex=start,additionalLabel="(ascending and descending)")

# virtual = g.RecordedData(UnityFolder+"ex3_test/stiar2_50_ours/1/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(boundary gaussian)")
# virtual = g.RecordedData(UnityFolder+"ex3_test/stiar2_50_ours/0/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(boundary gaussian)")
# virtual = g.RecordedData(UnityFolder+"ex3_test/stiar2_50_ours/2/",1).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel="(boundary gaussian)")

#virtual = g.RecordedData(folder+"recodingTest/late/",1).DrawHeadGraph(axes,startIndex=start,additionalLabel="(late boundary gaussian)")
#real = g.RecordedData(folder+"user3/임수빈/stair1_100/1/",2).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (real stair)",addtionalHeight=0.27,transX=16)
#real = g.RecordedData(folder+"user3/서승원/stair2_75/4/",2).DrawPosAndVelGraph(axes,startIndex=start,additionalLabel=" (real stair)",addtionalHeight=-0.5,transX=20)
#real = g.RecordedData(folder+"real/4/",2).DrawHeadGraph(axes,startIndex=start,additionalLabel=" (real stair)",addtionalHeight=-0.4,transX=0)

def saveFolder(height,bpm,method):
    stair = "stair1"
    if bpm == 1 :
        bpm = str(75)
    elif bpm ==2:
        bpm = str(100)
    if height == 2:
        stair = "stair2"

    return "experiment/" +stair + "_" + bpm + "_" + method + "/"

for i in range(1,3):
    g.RecordedData(UnityFolder + "ex3_test/stiar2_Ours/"+str(i) +"/", 1,False).writeToTxt(UnityFolder + saveFolder(2,i,"ours"))
    g.RecordedData(UnityFolder + "ex3_test/stiar2_Nagao/" + str(i) + "/", 1, False).writeToTxt(UnityFolder + saveFolder(2,i,"nagao"))
    g.RecordedData(UnityFolder + "ex3_test/stiar2_Seo/" + str(i) + "/", 1, False).writeToTxt(UnityFolder + saveFolder(2,i,"seo"))
    g.RecordedData(UnityFolder + "ex3_test/stiar1_Ours/"+str(i) +"/", 1,False).writeToTxt(UnityFolder + saveFolder(1,i,"ours"))
    g.RecordedData(UnityFolder + "ex3_test/stiar1_Nagao/" + str(i) + "/", 1, False).writeToTxt(UnityFolder + saveFolder(1,i,"nagao"))
    g.RecordedData(UnityFolder + "ex3_test/stiar1_Seo/" + str(i) + "/", 1, False).writeToTxt(UnityFolder + saveFolder(1,i,"seo"))

#g.RecordedData(UnityFolder + "test/0/",2).DrawPosAndVelGraph(axes)
g.RecordedData(UnityFolder + "user3/서승원/stair1_75/4/",2).DrawPosAndVelGraph(axes,addtionalHeight=0.6)
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

plt.show()

# xAxis = np.array(list(range(start, len(Rvelocity[1])))) * g.fixedDeltaTime
# axes[1].plot(xAxis,Rvelocity[1][start:],color="pink")
# axes[1].plot(xAxis,Lvelocity[1][start:],color="pink")
#virtual = g.RecordedData(folder+"recodingTest/sp4/",1).DrawPosAndVelGraph(axes)
#folder2 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
#real = g.RecordedData(folder2,2)
