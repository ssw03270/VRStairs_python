import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import GraphMaker as g
import TrajectorySplitter as spl
import os

folder = "D:/Desktop/unity/VRStair/footdata/ex3/our/임수빈/stair2_100/4/"
UnityFoloder = "D:/Desktop/unity/VRStair/footdata/"
ProjectFolder = os.getcwd()  #"C:/Users/user/Desktop/Unity/VRStair/footdata/"#"C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
ProjectFolder = ProjectFolder.replace("\\","/",10)
ProjectFolder += "/foot_dataset/"
def test():
    r = g.RecordedData(folder, firstZero=False)
    spl.FindPoints(r.RFootData.realPosData[1],NotRemove=False,isDebug=True)

def make(fileName,outFoloder,bpm,methodd,height):
    r = g.RecordedData(fileName,firstZero=False)
    f,axes = plt.subplots(2,1)

    #rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) ]
    yTrajectory = pd.read_csv("test.csv")
    r.DrawPosAndVelGraph(axes)
    y50 = yTrajectory[(yTrajectory["bpm"] == bpm) & (yTrajectory["method"] == methodd) & (yTrajectory["stairHeight"] == height)]
    r.HeightTrajectorySynthesize(np.array(y50["y"]),axes)

    if not os.path.exists(outFoloder):
        os.makedirs(outFoloder)

    s,e = r.FindStartAndEndIndex()
    axes[0].vlines(s * 0.011111 , 0,1,color="r")
    axes[0].vlines(e * 0.011111, 0, 1, color="b")

    r.writeToTxt1(outFoloder,max(s-10,0),e + 10)

    #r.DrawPosAndVelGraph(axes)
    plt.show()


def makeByOrder(fileName,outFoloder,bpm,methodd,height):
    r = g.RecordedData(fileName,firstZero=False)
    f,axes = plt.subplots(2,1)

    #rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) ]
    yTrajectory = pd.read_csv("test_order.csv")
    r.DrawPosAndVelGraph(axes)
    y_r0 = yTrajectory[(yTrajectory["bpm"] == bpm) & (yTrajectory["method"] == methodd) & (yTrajectory["stairHeight"] == height) & (yTrajectory["order"] == "r0")]
    y_r1 = yTrajectory[(yTrajectory["bpm"] == bpm) & (yTrajectory["method"] == methodd) & (yTrajectory["stairHeight"] == height) & (yTrajectory["order"] == "r1")]
    y_l0 = yTrajectory[(yTrajectory["bpm"] == bpm) & (yTrajectory["method"] == methodd) & (yTrajectory["stairHeight"] == height) & (yTrajectory["order"] == "l0")]
    r.HeightTrajectorySynthesizeByOrder(np.array(y_r0["y"]),np.array(y_r1["y"]),np.array(y_l0["y"]),axes)

    if not os.path.exists(outFoloder):
        os.makedirs(outFoloder)

    s,e = r.FindStartAndEndIndex()
    axes[0].vlines(s * 0.011111 , 0,1,color="r")
    axes[0].vlines(e * 0.011111, 0, 1, color="b")

    r.writeToTxt1(outFoloder,max(s-10,0),e + 10)

    #r.DrawPosAndVelGraph(axes)
    plt.show()


def makeOne(fileName,bpm):
    saveFolder = "/test/"
    methodList = ["Ours", "Nagao", "Seo"]
    for method in methodList:
        makeByOrder(ProjectFolder + fileName, UnityFoloder + saveFolder + "stair1_" + str(bpm) + "/" + method + "/", bpm,method,0.125)

def makeTwo(fileName,bpm):
    saveFolder = "/test/"
    methodList = ["Ours", "Nagao", "Seo"]
    for method in methodList:
        makeByOrder(ProjectFolder + fileName, UnityFoloder + saveFolder + "stair2_" + str(bpm) + "/" + method + "/", bpm,method,0.25)

def makeTrajectory1():
    fileList = ["ex3/Ours/임수빈/stair1_50/4/", "ex3/Ours/서승원/stair1_75/1/", "ex3/Ours/김미송/stair1_100/1/"]
    saveFolder = "/test/"
    bpmList = [50,75,100]
    methodList = ["Ours","Nagao","Seo"]
    for i,f in enumerate(fileList):
        for method in methodList:
            print(bpmList[i],method)
            makeByOrder(ProjectFolder+f,UnityFoloder + saveFolder + "stair1_" +str(bpmList[i]) + "/"+method + "/",bpmList[i],method,0.125)

def makeTrajectory2():
    fileList = ["ex3/Ours/임수빈/stair1_50/4/", "ex3/Ours/김미송/stair2_75/1/", "ex3/Seo/서승원/stair2_100/3/"]
    saveFolder = "/test/"
    bpmList = [50,75,100]
    methodList = ["Ours","Nagao","Seo"]
    for i,f in enumerate(fileList):
        for method in methodList:
            print(bpmList[i],method)
            makeByOrder(ProjectFolder+f,UnityFoloder + saveFolder + "stair2_" +str(bpmList[i]) + "/"+method + "/",bpmList[i],method,0.25)

#makeOne("ex3/our/김미송/stair1_100/1/",100,0.125)
#makeTwo("ex3/Seo/서승원/stair2_100/3/",100)
makeTrajectory1()
makeTrajectory2()
#test()