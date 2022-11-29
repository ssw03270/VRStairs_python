import string

import pandas

from define import *
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ProjectFolder = os.getcwd()  #"C:/Users/user/Desktop/Unity/VRStair/footdata/"#"C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
ProjectFolder = ProjectFolder.replace("\\","/",10)

ex2Folder = ProjectFolder + "/foot_dataset/ex2/"

MethodPerColor = {"Ours":"r","Seo": "g", "Nagao":"b"}

#f, axes = plt.subplots(3, 1)
def MakeVelData(posData,smoothON = False):
    velData = [0]
    if smoothON :
        posData = savgol_filter(posData, filterSize, 6)
    for i in range(1,len(posData)):
        velData.append((posData[i] - posData[i-1])/fixedDeltaTime)
    return velData.copy()

class Ex2Trajectory:
    StepId = 0
    def __init__(self,foloderName : str):
        self.folder = foloderName
        tIndex = self.folder[-5:-1].find("c")
        infoFolder = self.folder[:-5+tIndex] + "c" + str(int(self.folder[-5+tIndex+1:-1])+1) + "/"
        #print(infoFolder)
        f = open(infoFolder+ "info.txt" , 'r')
        info = f.read()
        f.close()
        self.stairHeight = float(info[info.find(":") + 1 : info.find(",")])
        self.bpm = float(info[info.find(",") + 5 : ])
        self.method = self.getMethod()
        self.name = self.getName()
        self.rFoot = pd.read_csv(self.folder + "RightFootController.csv",encoding='utf-8')
        self.lFoot = pd.read_csv(self.folder + "LeftFootController.csv",encoding='utf-8')
        self.head = pd.read_csv(self.folder + "Head.csv")
        self.rSplitPoints = []
        self.lSplitPoints = []
        self.Split()
        #self.MakeDataFrame()


    def MakeDataFrame(self):
        if len(self.rSplitPoints) > 0 and len(self.lSplitPoints) > 0 :
            rData : pandas.DataFrame = self.makeStepDataFrame(self.rSplitPoints,self.rFoot)
            lData : pandas.DataFrame = self.makeStepDataFrame(self.lSplitPoints, self.lFoot)
            return rData,lData
        else:
            return None,None

    def getNewDataFrame(self,df,s,e):
        newDf = df.loc[s:e].copy()
        newDf['bpm'] = self.bpm
        newDf['name'] = self.name
        newDf['method'] = self.method
        newDf['stairHeight'] = self.stairHeight
        newDf['StepId'] = Ex2Trajectory.StepId
        newDf['index'] = list(range(0,e-s+1))
        Ex2Trajectory.StepId += 1

        print(Ex2Trajectory.StepId)

        return newDf.copy()

    def makeStepDataFrame(self,splitPoints : list, df : pandas.DataFrame):
        totalDF = self.getNewDataFrame(df,splitPoints[0][0],splitPoints[0][1])
        for s,e in splitPoints[1:]:
            newdf = self.getNewDataFrame(df,s,e)
            totalDF = pd.concat([totalDF,newdf.copy()])

        return totalDF

    def getName(self):
        index = self.folder.find("ex2/")
        return self.folder[index + 4: index + 7]

    def getMethod(self):
        if "BoundaryGaussian" in self.folder:
            return "Ours"
        elif "ascendingDescending" in self.folder:
            return "Nagao"
        else:
            return "Seo"

    def findStartPoint(self,vel):
        mid = 0
        for p in range(70,len(vel)):
            if vel[p] > 0.5 and vel[p] < 2:
                mid = p
                break
        startIndex = 0
        for i in range(mid,0,-1):
            if vel[i] < 0.01:
                startIndex = i
                break
        return startIndex

    def findStartPointByPos(self,pos):
        #np.where(pos == max(pos))[0][0]
        maxH = max(pos)
        mid = 0
        for i in range(len(pos)):
            if pos[i] > maxH/2:
                mid = i
                break
        startIndex = 0
        if mid > 0:
            minH = min(pos[0:mid])
            for i in range(mid,0,-1):
                if pos[i] < minH + 0.005:
                    startIndex = i
                    break
            #print(mid, startIndex,pos[mid],minH)
        return startIndex


    def Split(self):
        d = np.array(self.rFoot["posY"])
        startI = self.findStartPoint(MakeVelData(d))
        self.rFoot["posY"] = savgol_filter(d - d[startI], filterSize, 6)
        self.rSplitPoints = self.splitTrajectory(np.array(self.rFoot["posY"]))

        d = np.array(self.lFoot["posY"])
        startI = self.findStartPoint(MakeVelData(d))
        self.lFoot["posY"] = savgol_filter(d - d[startI], filterSize, 6)
        self.lSplitPoints = self.splitTrajectory(np.array(self.lFoot["posY"]))

    def splitTrajectory(self,posData,isDebug = False):
        #posData = savgol_filter(posData, filterSize, 6)
        velData = MakeVelData(posData)
        aData = MakeVelData(velData,True)
        pList = self.GetPointList(posData,velData)

        # if self.stairHeight == 0.25 and self.bpm == 50:
        #     isDebug = True

        if isDebug:
            f, axes = plt.subplots(3, 1)
            axes[0].plot(posData, color=MethodPerColor[self.method], label=self.method)
            axes[1].plot(velData, color=MethodPerColor[self.method], label=self.method)
            axes[2].plot(aData, color=MethodPerColor[self.method], label=self.method)
            for p,e in pList:
                axes[0].scatter(p, posData[p])
                axes[0].scatter(e, posData[e])
                axes[1].scatter(p, velData[p])
                axes[1].scatter(e, velData[e])
            plt.show()
        return pList

    def isValidStep(self,data,Th):
        if len(data) > 140:
            print("cut:",len(data))
            return False
        elif len(data) < 60:
            return False
        if max(data) - min(data) < Th:
            return False
        if data[0] - data[-1] > Th * 0.5:
            return False
        else:
            return True

    def GetPointList(self,posData,velData):
        aData = MakeVelData(velData, True)
        i = 0
        Th = 0.05
        NextCoolTime = 25
        curCool = 0
        pointList = []
        maxV = 0
        for v in velData:
            if curCool < 0:
                if (abs(abs(v) - Th) < 0.02 and abs(aData[i]) < 2 and posData[i] < 0.1):
                    curCool = NextCoolTime
                    pointList.append(i)
            else:
                curCool -= 1
            i += 1
        resultList = []
        validTh = max(posData) * 0.6
        for j in range(len(pointList)-1):
            start = pointList[j]; end = pointList[j+1]
            start += self.findStartPointByPos(posData[start:end])
            if self.isValidStep(posData[start:end],validTh):
                resultList.append((start,end))

        return resultList


    def Draw(self):
        #sns.distplot(self.rFoot["posY"])

        key = "posY"

        plt.set_title(str.format("stairHeight : {0}, bpm : {1}",self.stairHeight,self.bpm))
        plt.plot(self.rFoot[key][1:] - self.rFoot[key][0],color = MethodPerColor[self.method],label = self.method)
        #plt.plot(self.lFoot[key][1:1000] - self.lFoot[key][0],color = MethodPerColor[self.method])
        #plt.plot(self.head[key][1:] - self.head[key][0])

        #plt.plot(self.rFoot["RotX"][1:]-self.rFoot["RotX"][0],self.rFoot["RotY"][1:] - self.rFoot["RotY"][0])
        # plt.plot(self.lFoot["RotY"][1:] - self.lFoot["RotY"][0])
        # plt.plot(self.head["RotY"][1:] - self.head["RotY"][0])
        #plt.show()


class TrajectorySet:
    def __init__(self,trajectoryFolder):
        self.case = dict()
        self.folders = self.GetTrajectoryFolders(trajectoryFolder)
        Ex2Trajectory.StepId = 0
        isFirst = True
        rDf = pd.DataFrame()
        lDf = pd.DataFrame()
        for f in self.folders:
            for i in range(0,17):
                t = Ex2Trajectory(f + "c"+str(i) + "/")
                curKey = (t.stairHeight,t.bpm,t.method)
                r,l = t.MakeDataFrame()
                if r is not None and l is not None :
                    rDf = pd.concat([rDf,r])
                    lDf = pd.concat([lDf,l])
                if(curKey in self.case.keys()):
                    self.case[curKey].append(t)
                else:
                    self.case[curKey] = [t]
        rDf.to_csv("R_test.csv",encoding='utf-8-sig')
        lDf.to_csv("L_test.csv",encoding='utf-8-sig')


    def GetTrajectoryFolders(self,trajectoryFolder):
        trajectoryFolderList = []
        if os.path.exists(trajectoryFolder):
            nameList = os.listdir(trajectoryFolder)
            methodList = ["BoundaryGaussian","ascendingDescending","pre2"]
            for name in nameList:
                for method in methodList:
                    curTrajectoryFoloder = trajectoryFolder + name + "/" + method + "/trajectory/"
                    if os.path.exists(curTrajectoryFoloder):
                        temp = os.listdir(curTrajectoryFoloder)
                        realFolder = curTrajectoryFoloder + "/0/"
                        mCount = 0
                        for t in temp:
                            l = len(os.listdir(curTrajectoryFoloder + t + "/"))
                            if l > mCount:
                                realFolder = curTrajectoryFoloder + t + "/"
                                mCount = l
                                if l == 18:
                                    trajectoryFolderList.append(realFolder)
                                    break
            print(trajectoryFolderList)
            return trajectoryFolderList

    def GetTrajectory(self,key):
        return self.case[key]

    def test(self,key):
        tdata = self.case[key]
        for t in tdata:
            t.Split()

    def Draw(self,key):
        tdata = self.case[key]
        for t in tdata:
            t.Draw()


#sets.test((0.25,100,"Nagao"))
#sets.Draw((0.25,100,"Ours"))
#sets.Draw((0.25,75,"Seo"))
#sets.Draw((0.125,50,"Nagao"))
# plt.legend()
# plt.show()

bpmList = [50,75,100]
heightList = [0.125,0.15,0.175,0.2,0.225,0.25]
methodList = ["Ours","Seo","Nagao"]
pNameList = ["강경은","이철우","이로운","김봉규","하창범"]


rData = pd.read_csv("R_test.csv",encoding='utf-8-sig')
lData = pd.read_csv("L_test.csv",encoding='utf-8-sig')


def MakeMeanHeightTrajectory(data,bpm,stairHeight,method):
    df = pd.DataFrame()
    df['y'] = data
    df['bpm'] = bpm
    df['method'] = method
    df['stairHeight']=stairHeight
    return df


def DrawPerParameter(height,method,bpm,person,axes):
    rDf = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) &  (rData["name"] == person)]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm) &  (lData["name"] == person)]
    stepList1 = rDf["StepId"].unique()
    stepList2 = lDf["StepId"].unique()
    for s in stepList1:
        data = rDf[rDf["StepId"] == s]
        d = np.array(data["posY"])

        axes.plot(list(np.arange(0, len(data["posY"]))), d ,color = MethodPerColor[method])

    for s in stepList2:
        data = lDf[lDf["StepId"] == s]
        d = np.array(data["posY"])
        axes.plot(list(np.arange(0, len(data["posY"]))), d, color=MethodPerColor[method])


def DrawPerParameterMean(height, method, bpm, axes):
    rDf = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) ]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm) ]
    cutLength = {50: 100, 75 : 75, 100 : 60}
    data = pd.concat([rDf,lDf],ignore_index=True).groupby('index').mean()
    d= np.array(data["posY"])

    for i in range(len(d)):
        if d[i] - d[0] < 0 or i > cutLength[bpm]:
            d[i] = d[0]
    d = savgol_filter(d - d[0], filterSize, 6)
    axes.plot(list(np.arange(0, len(data["posY"]))), d , color=MethodPerColor[method])
    return MakeMeanHeightTrajectory(d,bpm,height,method)
    #print(height,bpm,method,max(data["posY"]))
    # data = lDf.groupby('index').mean( )
    # d = np.array(data["posY"])
    # axes.plot(list(np.arange(0, len(data["posY"]))), d - d[0], color=MethodPerColor[method])

def DrawStairHeightAndMethod(height,method,bpm,axes):
    rDf = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm)]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm)]
    # stepList1 = rDf["StepId"].unique()
    # stepList2 = lDf["StepId"].unique()
    #sns.lineplot(x="index", y="posY", data=rDf, label=method)
    sns.lineplot(x="index", y="posY",ax = axes, data=pd.concat([rDf,lDf],ignore_index=True),label=method)
    # for s in stepList:
    #     data = h1[h1["StepId"] == s]
    #     d = np.array(data["posY"])
    #     plt.plot(list(np.arange(0, len(data["posY"]))), d -d[0], color=color)

def Draw(height,method,person,axes,color = None):
    rDf = rData[(rData["stairHeight"] == height) & (rData["name"] == person)]
    lDf = lData[(lData["stairHeight"] == height) & (lData["name"] == person)]
    # stepList1 = rDf["StepId"].unique()
    # stepList2 = lDf["StepId"].unique()
    #sns.lineplot(x="index", y="posY", data=rDf, label=method)
    sns.lineplot(x="index", y="posY",ax =axes,  hue = "method",data=lDf)

#Draw(0.2,"Ours","문석호")


def SaveTrajectoryPng():
    f, axes = plt.subplots(3, 6,sharex=True,sharey=True)
    df = pandas.DataFrame()
    for h in range(len(heightList)):
        axes[0][h].set_title(str.format("h : {0}", heightList[h]))
        print(h)
        for m in methodList:
            for i in range(0,3):
                newdf = DrawPerParameterMean(heightList[h], m, bpmList[i],axes[i][h])
                df = pd.concat([df,newdf])
                #DrawPerParameter(heightList[h], m, bpmList[i],pName,axes[i][h])
                #DrawStairHeightAndMethod(heightList[h], m, bpmList[i], axes[i][h])
                axes[i][h].set_xlim(0, 140)
                axes[i][h].set_ylim(-0.1, 0.8)
                axes[i][h].grid(True)
    df.to_csv("test.csv")
    axes[0][0].set_ylabel("bpm : 50")
    axes[1][0].set_ylabel("bpm : 75")
    axes[2][0].set_ylabel("bpm : 100")
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.15)
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('test.png',dpi=200)
    plt.show()

def SaveTrajectoryPngPerPerSon(pName):
    f, axes = plt.subplots(3, 6,sharex=True,sharey=True)
    for h in range(len(heightList)):
        axes[0][h].set_title(str.format("h : {0}", heightList[h]))
        print(h)
        for m in methodList:
            for i in range(0,3):
                DrawPerParameter(heightList[h], m, bpmList[i],pName,axes[i][h])
                #DrawStairHeightAndMethod(heightList[h], m, bpmList[i],pName, axes[i][h])
                axes[i][h].set_xlim(0, 140)
                axes[i][h].set_ylim(-0.1, 0.8)
                axes[i][h].grid(True)

    axes[0][0].set_ylabel("bpm : 50")
    axes[1][0].set_ylabel("bpm : 75")
    axes[2][0].set_ylabel("bpm : 100")
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.15)
    plt.gcf().set_size_inches(10, 5)
    plt.savefig(pName+'.png',dpi=200)
    plt.show()
    #plt.cla()
    # DrawStairHeightAndMethod(0.2, "Ours", 50, axes[0])
    # DrawStairHeightAndMethod(0.2, "Seo", 50, axes[0])
    # DrawStairHeightAndMethod(0.2, "Nagao", 50, axes[0])
    # DrawStairHeightAndMethod(0.2,"Ours",75,axes[1])
    # DrawStairHeightAndMethod(0.2,"Seo",75,axes[1])
    # DrawStairHeightAndMethod(0.2,"Nagao",75,axes[1])
    # DrawStairHeightAndMethod(0.2, "Ours", 100, axes[2])
    # DrawStairHeightAndMethod(0.2, "Seo", 100, axes[2])
    # DrawStairHeightAndMethod(0.2, "Nagao", 100, axes[2])


    #plt.show()


#sets = TrajectorySet(ex2Folder)
SaveTrajectoryPng()

# for PNAME in pNameList:
#      SaveTrajectoryPngPerPerSon(PNAME)
#DrawPerParameter("0.125","Nagao",50,"김봉규")

# f, axes = plt.subplots(3, 6,sharex=True,sharey=True)
# axes[0][0].grid(True)
# axes[0][0].set_title("test")
# DrawStairHeightAndMethod(0.2, "Nagao", 100, axes[0][0])
# DrawStairHeightAndMethod(0.2, "Nagao", 75, axes[1][0])
# plt.show()



