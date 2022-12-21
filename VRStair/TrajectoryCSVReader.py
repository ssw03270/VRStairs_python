
'''
Ex2 데이터를 분석하기 위한 코드
EX2 10칸씩 걸은 궤적 데이터를 csv 형식으로 저장해 둠. 이 csv를 분석하는 코드.

1) .csv가 모여있는 폴더 이름을 얻는다   -->  def GetTrajectoryFolders(self,trajectoryFolder):
    폴더 이름 및 info.txt를 통해 해당 csv의 사용자 이름,bpm, method, stair height 정보를 알 수 있음. --> Ex2Trajectory에 저장.

2) Ex2Trajectory : 10칸을 걷는 동안의 궤적 csv를 저장. 발걸음 분리를 해줌.
      - 이 발걸음(StepId)을 누가(name) , 어떤 bpm으로 , 어떤 stair height 를 , 어떤 방법으로 올랐는지를 csv로 저장함.
      - csv 형식 :   posX,posY,posZ,RotX,RotY,RotZ,RotW,vPosX,vPosY,vPosZ,initH,bpm,name,method,stairHeight,StepId,index

3) 그 발걸음(Step) 데이터들을 모아서 평균 높이 궤적을 구하고, csv로 저장.
     - csv 형식 : ,y,bpm,method,stairHeight

'''

import pandas
from define import *
from utility import *
from TrajectorySplitter import *
import GraphMaker as g
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ProjectFolder = os.getcwd()  #"C:/Users/user/Desktop/Unity/VRStair/footdata/"#"C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
ProjectFolder = ProjectFolder.replace("\\","/",10)

ex2Folder = ProjectFolder + "/foot_dataset/ex2/"
ex3Folder = ProjectFolder + "/foot_dataset/ex3/"

MethodPerColor = {"Ours":"r","Seo": "g", "Nagao":"b"}
bpmList = [50,75,100]
heightList = [0.125,0.15,0.175,0.2,0.225,0.25]
methodList = ["Ours","Seo","Nagao"]

global forStepCount

class Trajectory:
    StepId = 0
    def __init__(self,folderName : str,type="old"):
        self.folder = folderName
        self.stairHeight : float = 0
        self.bpm :int = 0
        self.method : str = "method"
        self.name : str = "name"
        self.rFoot : pd.DataFrame = pd.DataFrame()
        self.lFoot : pd.DataFrame = pd.DataFrame()
        #self.head =  pd.DataFrame()
        self.rSplitPoints = []
        self.lSplitPoints = []
        self.init(type)
        self.Split()

    def init(self):
        print("test")
        pass

    def MakeDataFrame(self):
        if len(self.rSplitPoints) > 0 and len(self.lSplitPoints) > 0 :
            rData : pandas.DataFrame = self.makeStepDataFrame(self.rSplitPoints,self.rFoot,"r")
            lData : pandas.DataFrame = self.makeStepDataFrame(self.lSplitPoints, self.lFoot,"l")
            return rData,lData
        else:
            return None,None

    def getNewDataFrame(self,df,s,e,order,rl :str):
        newDf = df.loc[s:e].copy()
        newDf['bpm'] = self.bpm
        newDf['name'] = self.name
        newDf['method'] = self.method
        newDf['stairHeight'] = self.stairHeight
        newDf['StepId'] = Trajectory.StepId
        newDf['index'] = list(range(0,e-s+1))
        newDf['order'] = rl + str(order)
        Trajectory.StepId += 1

        print(Trajectory.StepId)

        return newDf.copy()

    def makeStepDataFrame(self,splitPoints : list, df : pandas.DataFrame,rl : str):
        totalDF = pd.DataFrame()
        index = 0
        for s,e in splitPoints[0:]:
            newdf = self.getNewDataFrame(df,s,e,index,rl)
            index += 1
            totalDF = pd.concat([totalDF,newdf.copy()])

        return totalDF

    def findStartPoint(self,vel):
        mid = 0
        for p in range(20,len(vel)):
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
        startH = FindGroundHeight(d,MakeVelData(d)) #d[self.findStartPoint(MakeVelData(d))]
        self.rFoot["posY"] = savgol_filter(d - startH, filterSize, 6)
        self.rSplitPoints = self.splitTrajectory(np.array([self.rFoot["posX"],self.rFoot["posY"],self.rFoot["posZ"]]))

        d = np.array(self.lFoot["posY"])
        startH = FindGroundHeight(d, MakeVelData(d)) # d[self.findStartPoint(MakeVelData(d))]
        self.lFoot["posY"] = savgol_filter(d - startH, filterSize, 6)
        self.lSplitPoints = self.splitTrajectory(np.array([self.lFoot["posX"],self.lFoot["posY"],self.lFoot["posZ"]]))

    def splitTrajectory(self,posData,isDebug = False):
        #posData = savgol_filter(posData, filterSize, 6)
        velData = MakeVelData(posData[1])
        aData = MakeVelData(velData,True)
        self.isDebug = isDebug;
        pList = self.GetPointList(posData,velData).copy()
        # if self.stairHeight == 0.25 and self.bpm == 50 and self.name == "임수빈":
        #     isDebug = True


        if self.isDebug:
            f, axes = plt.subplots(3, 1)
            axes[0].plot(posData[1], color=MethodPerColor[self.method], label=self.method)
            axes[1].plot(posData[0], color=MethodPerColor[self.method], label=self.method)
            axes[2].plot(posData[2], color=MethodPerColor[self.method], label=self.method)
            #axes[2].plot(aData, color=MethodPerColor[self.method], label=self.method)
            for p,e in pList:
                axes[0].scatter(p, posData[1][p])
                axes[0].scatter(e, posData[1][e])
                axes[1].scatter(p, posData[0][p])
                axes[1].scatter(e, posData[0][e])
                axes[2].scatter(p, posData[2][p])
                axes[2].scatter(e, posData[2][e])
                # axes[1].scatter(p, velData[p])
                # axes[1].scatter(e, velData[e])
            plt.show()
        return pList

    def isValidStep(self,data,Th):
        if len(data[1]) > 140:
            print("cut:",len(data[1]))
            return False
        elif len(data[1]) < 60:
            return False
        if max(data[1]) - min(data[1]) < Th:
            return False
        if abs(data[0][0] - data[0][-1]) < 0.5:
            return False
        if abs(data[2][0] - data[2][-1]) > 0.3:
            return False
        if data[1][0] - data[1][-1] > Th * 0.5:
            return False
        else:
            print(abs(data[0][0] - data[0][-1]) )
            return True

    def GetPointList(self,posData,velData):
        aData = MakeVelData(velData, True)
        i = 0
        Th = 0.05
        NextCoolTime = 25
        curCool = 0
        pointList = []
        maxV = 0
        posYData = posData[1]
        for v in velData:
            if curCool < 0:
                if (abs(abs(v) - Th) < 0.02 and abs(aData[i]) < 2 and posYData[i] < 0.1):
                    curCool = NextCoolTime
                    pointList.append(i)
            else:
                curCool -= 1
            i += 1
        resultList = []
        validTh = max(posYData) * 0.6

        if(len(pointList) % 2 != 0):
            pointList.append(len(posYData)-1)

        for j in range(len(pointList)-1):
            start = pointList[j]; end = pointList[j+1]
            start += FindStartPoint(posYData[start:end])#self.findStartPointByPos(posData[start:end])
            if self.isValidStep(posData[:,start:end],validTh):
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

'''
Ex2 에서 기록한 csv 파일을 자르기 위한 class
'''
class Ex2Trajectory(Trajectory):
    def __init__(self,foloderName : str):
        super().__init__(foloderName)

    def init(self):
        tIndex = self.folder[-5:-1].find("c")
        infoFolder = self.folder[:-5+tIndex] + "c" + str(int(self.folder[-5+tIndex+1:-1])+1) + "/"
        f = open(infoFolder+ "info.txt" , 'r')
        info = f.read()
        f.close()
        self.stairHeight = float(info[info.find(":") + 1 : info.find(",")])
        self.bpm = float(info[info.find(",") + 5 : ])
        self.method = self.getMethod()
        self.name = self.getName()
        self.rFoot = pd.read_csv(self.folder + "RightFootController.csv",encoding='utf-8')
        self.lFoot = pd.read_csv(self.folder + "LeftFootController.csv",encoding='utf-8')
        #self.head = pd.read_csv(self.folder + "Head.csv")


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


class Ex2TrajectorySet:
    def __init__(self,trajectoryFolder):
        self.case = dict()
        self.folders = self.GetTrajectoryFolders(trajectoryFolder)

        Trajectory.StepId = 0
        isFirst = True
        rDf = pd.DataFrame()
        lDf = pd.DataFrame()
        global forStepCount
        forStepCount = 0
        for f in self.folders:
            for i in range(0,17):
                t = Ex2Trajectory(f + "c"+str(i) + "/")
                r,l = t.MakeDataFrame()
                if r is not None and l is not None :
                    rDf = pd.concat([rDf,r])
                    lDf = pd.concat([lDf,l])
                    print(t.name,":",len(rDf["StepId"].unique()) + len(lDf["StepId"].unique()))


        print("forStepCount", len(rDf["StepId"].unique()) + len(lDf["StepId"].unique()))
        rDf.to_csv("R_test.csv",encoding='utf-8-sig')
        lDf.to_csv("L_test.csv",encoding='utf-8-sig')

    def GetPointList(self,posData,velData):
        return FindPoints(posData[1])

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
                                    #print(realFolder)
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


'''
- 가상발 형식으로 되어있는 데이터의 발걸음 분리.
- 저장되어 있는 폴더 구조 : "ex3/method/사용자이름/계단높이_bpm/n" - 저장 되어 있음.
'''
class Trajectory_RD(Trajectory):
    def __init__(self,folderName, type = "old"):
        super().__init__(folderName,type)

    def init(self,type):
        self.infoDict = {}
        self.makeInfo()
        self.bpm = self.getBpm()
        self.stairHeight = self.getHeight()
        self.method = self.getMethod()
        self.name = self.getName()
        if type == "old":
            self.rFoot = g.RecordedFootData(self.folder + "RightFootController.txt").to_dataframe()
            self.lFoot =  g.RecordedFootData(self.folder + "LeftFootController.txt").to_dataframe()
        if type == "new":
            self.rFoot = pd.read_csv(self.folder + "RightFootController.csv", encoding='utf-8')
            self.lFoot = pd.read_csv(self.folder + "LeftFootController.csv", encoding='utf-8')

    def GetPointList(self,posData,velData):
        return FindPoints(posData[1])

    def makeInfo(self):
        key = ["ex","method","name","height","bpm"]
        index = self.folder.find("ex3/")
        value = self.folder[index:].split("/")
        self.infoDict = dict(zip(key,value))
        h,bpm = GetHeightAndBPM(self.infoDict["height"])
        self.infoDict["height"] = float(h)
        self.infoDict["bpm"] = int(bpm)

    def getName(self):
        return self.infoDict["name"]

    def getMethod(self):
        return self.infoDict["method"]

    def getHeight(self):
        return self.infoDict["height"]

    def getBpm(self):
        return self.infoDict["bpm"]

def MakeMeanTrajectoryByRecordedData(folderName):
    fList = GetFolderList(folderName)
    rDf = pd.DataFrame()
    lDf = pd.DataFrame()
    for f in fList:
        if ("조장현" in f) or ("김민정" in f) :
            continue
        if ("임수빈" in f ) or ("서승원" in f) or ("김미송" in f):
            t = Trajectory_RD(f)
        else:
            t = Trajectory_RD(f,"new")
        r, l = t.MakeDataFrame()
        if r is not None and l is not None:
            rDf = pd.concat([rDf, r])
            lDf = pd.concat([lDf, l])
    rDf.to_csv("R_test.csv",encoding='utf-8-sig')
    lDf.to_csv("L_test.csv",encoding='utf-8-sig')
    return


rData = pd.read_csv("R_test.csv",encoding='utf-8-sig')
lData = pd.read_csv("L_test.csv",encoding='utf-8-sig')


def MakeMeanHeightTrajectory(data,bpm,stairHeight,method):
    df = pd.DataFrame()
    df['y'] = data
    df['bpm'] = bpm
    df['method'] = method
    df['stairHeight'] = stairHeight

    return df

def MakeMeanHeightTrajectory_order(data,bpm,stairHeight,method,order):
    df = pd.DataFrame()
    df['y'] = data
    df['bpm'] = bpm
    df['method'] = method
    df['stairHeight'] = stairHeight
    df['order'] = order

    return df


def DrawPerParameter(height,method,bpm,person,axes):
    rDf = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) &  (rData["name"] == person)]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm) &  (lData["name"] == person)]
    data = pd.concat([rDf,lDf],ignore_index=True)
    sList = data["StepId"].unique()
    for s in sList:
        dd = data[data["StepId"] == s]
        d = np.array(dd["posY"])
        axes.plot(list(np.arange(0, len(dd["posY"]))), d ,color = MethodPerColor[method])

    return len(sList)

def DrawPerParameterByOrder(height, method, bpm, person,order, axes):
    rDf = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) & (rData["name"] == person) & (rData["order"] == order)]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm) & (lData["name"] == person) & (lData["order"] == order)]
    data = pd.concat([rDf, lDf], ignore_index=True)
    sList = data["StepId"].unique()
    for s in sList:
        dd = data[data["StepId"] == s]
        d = np.array(dd["posY"])
        axes.plot(list(np.arange(0, len(dd["posY"]))), d, color=MethodPerColor[method])

    return len(sList)

    # for s in stepList2:
    #     data = lDf[lDf["StepId"] == s]
    #     d = np.array(data["posY"])
    #     axes.plot(list(np.arange(0, len(data["posY"]))), d, color=MethodPerColor[method])

def DrawPerParameterXYZ(height,method,bpm,person,axes,i):
    rDf = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) &  (rData["name"] == person)]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm) &  (lData["name"] == person)]
    Df= pd.concat([rDf,lDf],ignore_index=True)
    stepList1 = Df["StepId"].unique()
    #stepList2 = lDf["StepId"].unique()
    #print(stepList1)
    for s in stepList1:
        data = Df[Df["StepId"] == s]
        dx = np.array(data["posX"])
        dy = np.array(data["posY"])
        dz = np.array(data["posZ"])
        axes[0][i].plot(list(np.arange(0, len(data["posX"]))), abs(dx-dx[0]) , color=MethodPerColor[method])
        axes[1][i].plot(list(np.arange(0, len(data["posY"]))), dy -dy[0],color = MethodPerColor[method])
        axes[2][i].plot(list(np.arange(0, len(data["posZ"]))), dz-dz[0], color=MethodPerColor[method])

def DrawPerParameterXYZMean(height,method,bpm,axes,i):
    rDf = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) ]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm) ]
    data = pd.concat([rDf, lDf], ignore_index=True)

    sList = data["StepId"].unique()
    for s in sList:
        dd = data[data["StepId"] == s]
        x = np.array(dd["posX"])
        x = abs(x - x[0])
        #for i in range(len(dd)):
        data.loc[(data["StepId"] == s),"posX"] = x
    data = data.groupby('index').mean()
    axes[0][i].plot(list(np.arange(0, len(data["posX"]))),data["posX"], color = MethodPerColor[method])
    axes[1][i].plot(list(np.arange(0, len(data["posY"]))),data["posY"], color = MethodPerColor[method])
    axes[2][i].plot(list(np.arange(0, len(data["posZ"]))),data["posZ"], color = MethodPerColor[method])


def DrawPerParameterMean(height, method, bpm, axes):
    rDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm)]
    lDf = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm)]
    cutLength = {50: 100, 75 : 75, 100 : 60}
    data = pd.concat([rDf,lDf],ignore_index=True).groupby('index').mean()
    d= np.array(data["posY"])
    for i in range(len(d)):
        if d[i] - d[0] < 0 or i > cutLength[bpm]:
            d[i] = d[0]
    d = savgol_filter(d - d[0], filterSize, 6)
    axes.plot(list(np.arange(0, len(data["posY"]))), d , color=MethodPerColor[method])
    return MakeMeanHeightTrajectory(d,bpm,height,method)


def DrawPerParameterMean_order(height, method, bpm, order,axes):
    cutLength = {50: 100, 75 : 75, 100 : 65}
    if "r" in order:
        data = rData[(rData["stairHeight"] == height) & (rData["method"] == method) & (rData["bpm"] == bpm) & (rData["order"] == order)]
    else:
        data = lData[(lData["stairHeight"] == height) & (lData["method"] == method) & (lData["bpm"] == bpm) & (lData["order"] == order)]

    data = data.groupby('index').mean()
    d = np.array(data["posY"])
    for i in range(len(d)):
        if d[i] - d[0] < 0 or i > cutLength[bpm]:
            d[i] = d[0]
    d = savgol_filter(d - d[0], filterSize, 6)
    axes.plot(list(np.arange(0,len(d))), d , color=MethodPerColor[method])
    return MakeMeanHeightTrajectory_order(d,bpm,height,method,order)


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

def SaveTrajectoryPngPerPerSonByOrder(pName):
    f, axes = plt.subplots(3, 6,sharex=True,sharey=True)
    hList = [0.125,0.25]
    for h in range(len(hList)):
        print(h)
        for m in methodList:
            for i in range(0,3):
                for k,o in enumerate(["r0","r1","l0"]):
                    axes[0][3*h + k].set_title(str.format("h : {0},order: {1}", hList[h],o))
                    DrawPerParameterByOrder(hList[h], m, bpmList[i],pName,o,axes[i][3*h + k])
                    axes[i][3*h + k].set_xlim(0, 140)
                    axes[i][3*h + k].set_ylim(-0.1, 0.8)
                    axes[i][3*h + k].grid(True)

    axes[0][0].set_ylabel("bpm : 50")
    axes[1][0].set_ylabel("bpm : 75")
    axes[2][0].set_ylabel("bpm : 100")
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.15)
    plt.gcf().set_size_inches(10, 5)
    plt.savefig(pName+'.png',dpi=200)
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

def SaveTrajectoryPngPerPerSonXYZ(pName):
    f, axes = plt.subplots(3, 6,sharex=True,sharey=True)
    for h in range(len(heightList)):
        axes[0][h].set_title(str.format("h : {0}", heightList[h]))
        print(h)
        for m in methodList:
            #for i in range(0,3):
            DrawPerParameterXYZ(heightList[h], m, bpmList[1], pName, axes,h)
            # axes[i][h].set_xlim(0, 140)
            # axes[i][h].set_ylim(-0.1, 0.8)
            # axes[i][h].grid(True)

    axes[0][0].set_ylabel("X")
    axes[1][0].set_ylabel("Y")
    axes[2][0].set_ylabel("Z")
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.15)
    plt.show()

def SaveTrajectoryPngXYZMean():
    f, axes = plt.subplots(3, 6,sharex=True,sharey=True)
    for h in range(len(heightList)):
        axes[0][h].set_title(str.format("h : {0}", heightList[h]))
        print(h)
        for m in methodList:
            #for i in range(0,3):
            DrawPerParameterXYZMean(heightList[h], m, bpmList[0], axes,h)
            # axes[i][h].set_xlim(0, 140)
            # axes[i][h].set_ylim(-0.1, 0.8)
            # axes[i][h].grid(True)

    axes[0][0].set_ylabel("X")
    axes[1][0].set_ylabel("Y")
    axes[2][0].set_ylabel("Z")
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.15)
    plt.show()

def SaveTrajectory1():
    f, axes = plt.subplots(3, 6,sharex=True,sharey=True)
    df = pandas.DataFrame()
    hList = [0.125,0.25]
    for h in range(len(hList)):
        print(h)
        for m in methodList:
            for i in range(0,3):
                axes[0][h].set_title(str.format("h : {0}", hList[h]))
                newdf = DrawPerParameterMean(hList[h], m, bpmList[i],axes[i][h])
                df = pd.concat([df,newdf])
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


def SaveTrajectory_order():
    f, axes = plt.subplots(3, 7,sharex=True,sharey=True)
    df = pandas.DataFrame()
    hList = [0.125,0.25]
    for h in range(len(hList)):
        print(h)
        for m in methodList:
            for i in range(0,3):
                for k,o in enumerate(['r0','r1','l0']):
                    axes[0][h*3+k].set_title(str.format("h : {0},order : {1}", hList[h],o))
                    newdf = DrawPerParameterMean_order(hList[h], m, bpmList[i],o,axes[i][h*3+k])
                    df = pd.concat([df,newdf])
                    axes[i][h*3+k].set_xlim(0, 140)
                    axes[i][h*3+k].set_ylim(-0.1, 0.8)
                    axes[i][h*3+k].grid(True)

    df.to_csv("test_order.csv")
    axes[0][0].set_ylabel("bpm : 50")
    axes[1][0].set_ylabel("bpm : 75")
    axes[2][0].set_ylabel("bpm : 100")
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.15)
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('test_order.png',dpi=200)
    plt.show()


def debugStep(dubugList):
    data = pd.concat([rData, lData], ignore_index=True)
    for s in dubugList:
        dd = data[(data["StepId"] == s) & (data["index"] == 0)]
        #print(dd["bpm"],dd["name"],dd["stairHeight"],dd["method"])


#SaveTrajectoryPngXYZMean()
if __name__ == "__main__":
    forStepCount = 0
    MakeMeanTrajectoryByRecordedData(ex3Folder)
    SaveTrajectory_order()
    #sets = Ex2TrajectorySet(ex2Folder)
    #SaveTrajectoryPng()
    # debugList = list(range(0,2488))
    # for PNAME in rData["name"].unique():
    #     print(PNAME)
    #     SaveTrajectoryPngPerPerSonByOrder(PNAME)
    # print(forStepCount)
    # #print(debugList)
    #debugStep(debugList)
# f,a = plt.subplots(2,1)
# DrawPerParameter("0.125","Nagao",75,"강경은",a)
# plt.show()


