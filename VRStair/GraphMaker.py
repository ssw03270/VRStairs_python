#수빈
#발 움직임 데이터 그래프


import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt

#folder = "blendingData/0722-compare/"
folder = "blendingData/realStair/"
#folder = "blendingData/"
rFootName = "RightFootController_"
lFootName = "LeftFootController_"
realName = "realTrajectory"
blendName = "blendedTrajectory"

def loadPosData(flieName):
    f = open(flieName, 'r')
    # 첫번째 pos 의 x,z값을 0으로 해줌.
    line = f.readline()
    pX = []
    pY = []
    pZ = []
    firstVec = []
    line = f.readline()
    line = line.replace("(", "").replace(")", "").replace(",", "")
    line = line.split()
    firstVec.append(float(line[0]))
    firstVec.append(float(line[1]))
    firstVec.append(float(line[2]))
    pX.append(0)
    pY.append(0)
    pY.append(0)
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        pX.append(float(line[0]))
        #if(float(line[1]) > -1 and float(line[1]) < 1.5) :
        pY.append(float(line[1]) - firstVec[1])
        pZ.append(float(line[2]))
    f.close()
    data = []
    data.append(pX);
    data.append(pY);
    data.append(pZ);
    return data

def loadData(flieName):
    f = open(flieName, 'r')
    # 첫번째 pos 의 x,z값을 0으로 해줌.
    line = f.readline()
    pX = []
    pY = []
    pZ = []
    line = f.readline()
    line = line.replace("(", "").replace(")", "").replace(",", "")
    line = line.split()
    while True:
        line = f.readline()
        if not line: break
        line = line.replace("(", "").replace(")", "").replace(",", "")
        line = line.split()
        pX.append(float(line[0]))
        #if(float(line[1]) > -1 and float(line[1]) < 1.5) :
        pY.append(float(line[1]))
        pZ.append(float(line[2]))
    f.close()
    data = []
    data.append(pX);
    data.append(pY);
    data.append(pZ);
    return data

def makeVectorData(data):
    vecData = []
    px = []
    py = []
    pz = []
    firstVec = []
    for d in data:
        d = d.replace("(", "").replace(")", "").replace(",", "")
        vec = d.split()
        if(len(px) == 0):
            px.append(0)
            py.append(0)
            pz.append(0)
            firstVec.append(float(vec[0]))
            firstVec.append(float(vec[1]))
            firstVec.append(float(vec[2]))
            continue
        if len(vec) == 3:
            px.append(float(vec[0]) - firstVec[0])
            py.append(float(vec[1])-firstVec[1])
            pz.append(float(vec[2]) - firstVec[2])
    vecData.append(px)
    vecData.append(py)
    vecData.append(pz)
    return vecData


class RecordedFootData():
    def __init__(self,fileName):
        self.realPosData = []
        self.realRotData = []
        self.blendPosData = []
        self.trackerHeightData = []
        self.LoadFootData(fileName)

    def LoadFootData(self,fileName):
        f = open(fileName, 'r')
        data = f.read()
        # 현실 위치, 회전, 조작된 위치, 트래커 높이
        dataList = data.split("####\n")
        self.realPosData = makeVectorData(dataList[0].split('\n'))
        self.realRotData = makeVectorData(dataList[1].split('\n'))
        self.blendPosData = makeVectorData(dataList[2].split('\n'))
        thd = dataList[3].split()
        for t in thd:
            self.trackerHeightData.append(float(t))
        f.close()


def ReadAndDrawGraph(pathArray):
    data = []
    for path in pathArray:
        data.append(loadPosData(path).copy())
    f, axes = plt.subplots(int(len(data)/2), 1)
    for i in range(int(len(data)/2)):
        axes[i].set_title(i)
        axes[i].plot(data[(i*2)][1],label = "L")
        axes[i].plot(data[(i*2)+1][1], label = "R")
    f.legend(loc='upper right')
    plt.show()

def ReadAndDrawGraph2(pathArray,axes,labelName = "compare",LColor = "indigo", RColor = "gold"):
    data = []
    for path in pathArray:
        data.append(RecordedFootData(path))
    axes[0].plot(data[0].blendPosData[1],LColor,label= labelName + "(L)");
    axes[0].plot(data[1].blendPosData[1], RColor,label= labelName + "(R)");
    axes[1].plot(data[0].realPosData[1],LColor);
    axes[1].plot(data[1].realPosData[1], RColor);

    axes[2].plot(data[2].blendPosData[1],LColor);
    axes[2].plot(data[3].blendPosData[1], RColor);
    axes[3].plot(data[2].realPosData[1],LColor);
    axes[3].plot(data[3].realPosData[1], RColor);


def ReadAndDrawGraph3(LeftPath,RightPath,axes,startIndex = 0,labelName = "compare",LColor = "indigo", RColor = "gold"):
    data = []
    data.append(RecordedFootData(LeftPath))
    data.append(RecordedFootData(RightPath))
    axes[startIndex * 2].plot(data[0].blendPosData[1],LColor,label= labelName + "(L)");
    axes[startIndex * 2].plot(data[1].blendPosData[1], RColor,label= labelName + "(R)");
    axes[startIndex * 2 + 1].plot(data[0].realPosData[1],LColor);
    axes[startIndex * 2 + 1].plot(data[1].realPosData[1], RColor);


def DrawRealStairGraph(axes,istwo = False,L=0,R = 1):
    stairPath = "blendingData/realStair/"
    if istwo:
        #d1 = loadPosData(folder+"Lfootdata2.txt")
        #d2 = loadPosData(folder+"Rfootdata2.txt")
        #d3 = loadPosData(folder+"Lfootdata1.txt")
        #d4 = loadPosData(folder +"Rfootdata1.txt")
        d1 = loadPosData(folder +"1/"+ "Lfootdata.txt")
        d2 = loadPosData(folder +"1/"+ "Rfootdata.txt")
        d3 = loadPosData(folder +"2/"+ "Lfootdata.txt")
        d4 = loadPosData(folder +"2/"+ "Rfootdata.txt")
    else:
        d1 = loadPosData(folder+"Lfootdata5.txt")
        d2 = loadPosData(folder+"Rfootdata5.txt")
        d3 = loadPosData(folder+"Lfootdata4.txt")
        d4 = loadPosData(folder+"Rfootdata4.txt")

    axes[L].plot(d2[1],'C0', label="real stair(L)");
    axes[L].plot(d1[1], 'C1',label="real stair(R)");
    axes[R].plot(d4[1],'C0');
    axes[R].plot(d3[1], 'C1');

def DrawCompareGraph():
    return


# f, axes = plt.subplots(2, 1)
# ReadAndDrawGraph3("blendingData/towerStair/L_new.txt","blendingData/towerStair/R_new.txt",axes,"new","C0","C1")
# ReadAndDrawGraph3("blendingData/towerStair/L_t4.txt","blendingData/towerStair/R_t4.txt",axes,"pre","C2","C3")
#
# f.legend(loc='upper right')
# plt.show()


#ReadAndDrawGraph([folder + "test/Lfootdata1.txt",folder + "test/Rfootdata1.txt",folder + "test/Lfootdata2.txt",folder + "test/Rfootdata2.txt" ])

#
# f1 = RecordedFootData(folder+"left_no_one_L.txt")
# f2 = RecordedFootData(folder+"right_no_one_L.txt")
# f5 = RecordedFootData(folder+"left_no_one_R.txt")
# f6 = RecordedFootData(folder+"right_no_one_R.txt")

f3 = RecordedFootData(folder+"upForce/"+"Left_L.txt") #왼발 궤적 - 왼발 먼저
f4 = RecordedFootData(folder+"upForce/"+"Right_L.txt") #오른발 궤적 - 왼발먼저
f7 = RecordedFootData(folder+"upForce/"+"Left_R.txt") # 왼발 궤적- 오른발먼저
f8 = RecordedFootData(folder+"upForce/"+"Right_R.txt") # 오른발 궤적 - 오른발 먼저


f, axes = plt.subplots(4, 1)

#ReadAndDrawGraph3("blendingData/terrain/L1.txt","blendingData/terrain/R1.txt",axes,0,"3","C0","C1")
ReadAndDrawGraph3("blendingData/terrain/L2.txt","blendingData/terrain/R2.txt",axes)
ReadAndDrawGraph3("blendingData/terrain/L15.txt","blendingData/terrain/R15.txt",axes,0,"new","C0","C1")
#ReadAndDrawGraph3("blendingData/terrain/L13.txt","blendingData/terrain/R13.txt",axes,0,"new","C2","C3")

'''
DrawRealStairGraph(axes,True,L = 0, R = 2)
ReadAndDrawGraph2([folder+"left_no_two_L.txt",folder+"right_no_two_L.txt", folder+"left_no_two_R.txt",folder+"right_no_two_R.txt"],axes,"pre")

axes[0].set_title('Real stair trajectory & virtual blended trajectory (first step : L-foot)')
axes[0].plot(f3.blendPosData[1],'C3',label = "new (L)");
axes[0].plot(f4.blendPosData[1],'C4',label = "new (R)");

axes[1].set_title('input trajectory of virtual trajectory (first step : L-foot)')
axes[1].plot(f3.realPosData[1],'C3');
axes[1].plot(f4.realPosData[1],'C4');

axes[2].set_title('Real stair trajectory & virtual blended trajectory (first step : R-foot)')
axes[2].plot(f7.blendPosData[1],'C3');
axes[2].plot(f8.blendPosData[1],'C4')

axes[3].set_title('input trajectory of virtual trajectory (first step : R-foot)')
axes[3].plot(f7.realPosData[1],'C3');
axes[3].plot(f8.realPosData[1],'C4')


#axes[3].plot(f7.realPosData[1],label = "real trajectory(no L)");
#axes[3].plot(f8.realPosData[1],label = "real trajectory(no R)");
# f9 = RecordedFootData(folder+"LeftFootController_one_R.txt")
# f10 = RecordedFootData(folder+"RightFootController_one_R.txt")
# f11 = RecordedFootData(folder+"LeftFootController_one_L.txt")
# f12 = RecordedFootData(folder+"RightFootController_one_L.txt")
f9 = RecordedFootData(folder+"left_no_one_R.txt")
f10 = RecordedFootData(folder+"right_no_one_R.txt")
f11 = RecordedFootData(folder+"left_no_one_L.txt")
f12 = RecordedFootData(folder+"right_no_one_L.txt")
#axes[0].plot(f11.blendPosData[1],label = "virtual stair(L)");
#axes[0].plot(f12.blendPosData[1],label = "virtual stair(R)");
axes[1].plot(f11.realPosData[1],'C5',label = "one step trajectory(L)");
axes[1].plot(f12.realPosData[1],'C6',label = "one step trajectory(R)");
#axes[2].plot(f9.blendPosData[1],label = "virtual stair(L)");
#axes[2].plot(f10.blendPosData[1],label = "virtual stair(R)");
axes[3].plot(f9.realPosData[1],'C5');
axes[3].plot(f10.realPosData[1],'C6');

waistData1 = np.array(loadData(folder +"1/WaistData.txt"))
waistData2 = np.array(loadData(folder +"2/WaistData.txt"))
w1 = np.array(loadData(folder +"1/"+ "Rfootdata.txt")) - waistData1
w2 = np.array(loadData(folder +"1/"+ "Lfootdata.txt")) - waistData1
axes[1].plot(w1[1]-w1[1][0],'C7',label = "waist-foot(L)")
axes[1].plot(w2[1]-w2[1][0],'C8',label = "waist-foot(R)")

w3 = np.array(loadData(folder +"2/"+ "Rfootdata.txt")) - waistData2
w4 = np.array(loadData(folder +"2/"+ "Lfootdata.txt")) - waistData2
axes[3].plot(w3[1]-w3[1][0],'C7')
axes[3].plot(w4[1]-w4[1][0],'C8')


f.legend(loc='upper right')
plt.show()

f1 = RecordedFootData()
f2 = RecordedFootData()
#f1.LoadFootData(folder+"LeftFootController(2022-07-24_03-22-02).txt")
#f2.LoadFootData(folder+"RightFootController(2022-07-24_03-22-02).txt")
#f2.LoadFootData(folder+"LeftFootController(2022-07-24_03-33-00).txt")
#f1.LoadFootData(folder+"RightFootController(2022-07-24_03-33-00).txt")
#f2.LoadFootData(folder+"LeftFootController(2022-07-24_03-49-11).txt")
#f1.LoadFootData(folder+"RightFootController(2022-07-24_03-49-11).txt")
f2.LoadFootData(folder+"LeftFootController(2022-07-24_03-52-48).txt")
f1.LoadFootData(folder+"RightFootController(2022-07-24_03-52-48).txt")

f, axes = plt.subplots(2, 1)
axes[0].set_title('blended trajectory')
axes[0].plot(f1.blendPosData[1],label = "blended(predict)");
axes[0].plot(f2.blendPosData[1],'r',label = "blended(not predict)");


axes[1].set_title('real trajectory')
axes[1].plot(f1.realPosData[1],label = "real(predict)");
axes[1].plot(f2.realPosData[1],label = "real(not predict)");

'''

f.legend(loc='upper right')
plt.show()


''' 경사, 계단 비교
f1 = RecordedFootData()
f2 = RecordedFootData()
f1.LoadFootData(folder+"LeftFootController(2022-07-24_00-21-29).txt")
f2.LoadFootData(folder+"RightFootController(2022-07-24_00-21-29).txt")

f3 = RecordedFootData()
f4 = RecordedFootData()
f3.LoadFootData(folder+"LeftFootController(2022-07-24_00-26-23).txt")
f4.LoadFootData(folder+"RightFootController(2022-07-24_00-26-23).txt")


f, axes = plt.subplots(2, 2)
axes[0][0].set_title('Stair(L)')
axes[0][0].plot(f1.realPosData[1],'b',label = "real position(L)");
axes[0][0].plot(f1.blendPosData[1],'r',label = "Blending (L)");


axes[1][0].set_title('Stair(R)')
axes[1][0].plot(f2.realPosData[1],'b',label = "real position(R)");
axes[1][0].plot(f2.blendPosData[1],'r',label = "Blending (R)");


axes[0][1].set_title('Slope(L)')
axes[0][1].plot(f3.realPosData[1],'b',label = "real position(L)");
axes[0][1].plot(f3.blendPosData[1],'r',label = "Blending (L)");


axes[1][1].set_title('Slope(R)')
axes[1][1].plot(f4.realPosData[1],'b',label = "real position(R)");
axes[1][1].plot(f4.blendPosData[1],'r',label = "Blending (R)");


f.legend(loc='upper right')
plt.show()

'''
'''
LrealPos = loadPosData(folder+lFootName+realName+"_PUb.txt");
LpPos = loadPosData(folder+lFootName+blendName+"_PUb.txt");
Lp_stair_real = loadPosData(folder+"/stair/"+lFootName+realName+"_PUb.txt");
Lp_stair_blend = loadPosData(folder+"/stair/"+lFootName+blendName+"_PUb.txt");

RrealPos = loadPosData(folder+rFootName+realName+"_PUb.txt");
RpPos = loadPosData(folder+rFootName+blendName+"_PUb.txt");
Rp_stair_real = loadPosData(folder+"/stair/"+rFootName+realName+"_PUb.txt");
Rp_stair_blend = loadPosData(folder+"/stair/"+rFootName+blendName+"_PUb.txt");


f, axes = plt.subplots(4, 1)
axes[0].plot(LrealPos[1],'b',label = "real position(L)");
axes[0].plot(LpPos[1],'r',label = "Blending (L)");
axes[1].plot(Lp_stair_real[1],label = "real position stair(L)");
axes[1].plot(Lp_stair_blend[1],label = "Blending stair(L)");



axes[2].plot(RrealPos[1],'g',label = "real position(R)");
axes[2].plot(RpPos[1],'m',label = "Blending1(R)");
axes[3].plot(Rp_stair_real[1],label = "real position stair(L)");
axes[3].plot(Rp_stair_blend[1],label = "Blending stair(L)");


# plt.plot(rp2 - rp1);
# plt.plot(rp4 - rp3);
f.legend(loc='upper right')
plt.show()
'''

'''
plt.plot(LrealPos[1],'b',label = "real position(L)");
plt.plot(LpPos[1],'r',label = "Blending1(L)");
plt.plot(LnpPos[1],'y',label = "Blending2(L)");


plt.plot(RrealPos[1],'g',label = "real position(R)");
plt.plot(RpPos[1],'m',label = "Blending1(R)");
plt.plot(RnpPos[1],'violet',label = "Blending2(R)");
#plt.plot(RnpPos[1],'violet',label = "not predict Blending(R)");



# plt.plot(rp2 - rp1);
# plt.plot(rp4 - rp3);
plt.legend(loc='upper right')
plt.show()
'''
