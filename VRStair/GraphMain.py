
import GraphMaker as g
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
#folder = "C:/Users/Dobby/Documents/GitHub/VRStairs_python/VRStair/foot_dataset/"
#folder = "C:/Users/user/Desktop/Unity/VRStair/footdata/"#"C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
#folder = "D:/Desktop/unity/VRStair/footdata/" #"C:/Users/user/Desktop/Unity/VRStair/footdata/"
ProjectFolder = os.getcwd()  #"C:/Users/user/Desktop/Unity/VRStair/footdata/"#"C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
ProjectFolder = ProjectFolder.replace("\\","/",10)
folder = ProjectFolder + "/foot_dataset/"
#folder = "C:/Users/user/Desktop/VRStair_python/git/VRStair/foot_dataset/"
#data =  g.RecordedData(folder)


#data.DrawGrahp(x = "Distance")
#data1.DrawGrahp()
#data2.DrawGrahp()

def writeCSV(resultDict,condition,name = "avg"):
    with open(name+".csv",'w',encoding="UTF-8",newline="") as f:
        w = csv.writer(f)
        order = g.StepAnalyzer.order#["Head 1","Head 2", "First Foot", "Second Foot", "Last Foot"]
        for c in condition:
            curInfo = resultDict[c].GetResultList()
            print(resultDict[c].GetResultList())
            #w.writerow([c])
            w.writerow([c, ""] + list(resultDict[condition[0]].GetResultList()[0][0].keys()))
            i = 0
            for o in order:
                w.writerow([o])
                avg = list(curInfo[0][i].values())
                sd = list(curInfo[1][i].values())
                w.writerow(["","avg"]+avg)
                w.writerow(["","SD"]+sd)
                i += 1
        print(name+".csv" +" is saved at")

'''
input
 - folderName : 경로(str)
 - condition : 계단 속도 및 크기 분류 [str,str,...,] ex)   ["stair2_75", "stair2_100"]
output
 -> result : 각 condittion을 key, StepAnalyzer를 value로 갖는 dict.
'''
def readFolder(folderName,condition):
    result = dict()
    for c in condition:
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0, 10):
            for name in file_list:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i) + "/")
        result[c] = g.StepAnalyzer(stepFiles, False, c)
    return result


def WriteDataFrame(folderName, condition = ["stair1_50", "stair1_75", "stair1_100", "stair2_50", "stair2_75", "stair2_100"]):
    result : g.StepAnalyzer = readFolder(folderName,condition)
    for c in result.keys():
        curDataFrameDict = result[c].dataFrameDict
        for key in curDataFrameDict.keys():
            curF = "dataFrame/"+ c +"/"
            os.makedirs(curF,exist_ok=True)
            curDataFrameDict[key].to_csv(curF + key +".csv")
            print("write DataFrame in ", curF)
    return


def MakeHeadDataFrame(folderName):
    condition = ["stair1_50", "stair1_75", "stair1_100", "stair2_50", "stair2_75", "stair2_100"]
    result  = readFolder(folderName,condition)
    for c in condition:
        df = result[c].MakeHeadFullTrajectoryData()
        curF = "dataFrame/"+ c + "/"
        os.makedirs(curF, exist_ok=True)
        df.to_csv(curF + "HeadFull.csv")
        #sns.lineplot(x="time", y="velY", data=df,label = c)

def MakeAllTrajectoryDataFrame(folderName,method= ""):
    condition = ["stair1_50", "stair1_75", "stair1_100","stair2_50", "stair2_75", "stair2_100"]
    result  = readFolder(folderName,condition)
    for c in condition:
        curF = "dataFrame/"+method + "/" + c + "/"
        df = result[c].MakeAllTrajectoryToCSV(curF)


def reader(folderName):
    condition = ["stair1_50", "stair1_75", "stair1_100", "stair2_50", "stair2_75", "stair2_100"]
    result = readFolder(folderName,condition)
    writeCSV(result,condition,"ex1_result")

def reader1(folderName):
    condition = ["stair1_60","stair2_60","stair1_85","stair2_85"]
    for c in condition:
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0,3):
            stepFiles.append(folderName + "서승원" + "/" + c + "/" + str(i)+ "/")
            stepFiles.append(folderName + "임수빈" + "/" + c + "/" + str(i) + "/")
        print(c)
        g.StepAnalyzer(stepFiles,False)

def reader2(folderName):
    result = dict()
    condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]
    file_list = os.listdir(folderName)
    for name in file_list:
        stepFiles = []
        for i in range(0,10):
            for c in condition:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i)+ "/")
                result[c] = g.StepAnalyzer(stepFiles,False,c).GetResultList()
        writeCSV(result,condition,name)

def Compare2Result(avgDict1 ,avgDict2):
    order = ["Head 1", "Head 2", "First Foot", "Second Foot", "Last Foot","Net speed(second)","Net speed(last)"]
    for i in range(len(avgDict1)):
        print("--------------------",order[i], "---------------------------------")
        print("< diff >")
        for k in avgDict1[i].keys():
            print(k ,":" ,avgDict1[i][k]-avgDict2[i][k],end=",")
        print("\n< 2/1 > : ")
        for k in avgDict1[i].keys():
            print(k ,":" ,avgDict2[i][k]/avgDict1[i][k],end=",")
        print()

def lengthCompare(folderName):
    result = dict()
    #condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]
    f,axes = plt.subplots(2,1)
    condition = ["stair1_50", "stair1_75", "stair1_100", "stair2_50", "stair2_75", "stair2_100"]
    num = 0
    for c in condition:
        file_list = os.listdir(folderName)
        stepFiles = []
        halfFiles = []
        halfName = c.replace("_", "_half_")
        for i in range(0,10):
            for name in file_list:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i)+ "/")
                halfFiles.append(folderName + name + "/" + halfName + "/" + str(i) + "/")
        print(c)
        sA = g.StepAnalyzer(stepFiles,False,c)
        sA_half = g.StepAnalyzer(halfFiles,False,halfName)
        result[c] = sA.GetResultList()
        sA.DrawLengthPerAscent(axes,"C"+str(num),num,c)
        sA_half.DrawLengthPerAscent(axes, "C" + str(num), num, halfName,marker="s")
        axes[0].legend()
        num += 1

def lengthCompare1(folderName):
    result = dict()
    #condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]
    f,axes = plt.subplots(2,1)
    condition = ["stair1_50","stair1_75","stair1_100"]
    condition1 = ["stair2_50", "stair2_75","stair2_100"]
    num = 0
    stepFiles = []
    for c in condition:
        file_list = os.listdir(folderName)
        for i in range(0,10):
            for name in file_list:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i)+ "/")
    sA = g.StepAnalyzer(stepFiles,False,c)
    sA.DrawLengthPerAscent1(axes,"C"+str(num),num,c)
    stepFiles = []
    for c in condition1:
        file_list = os.listdir(folderName)
        for i in range(0,10):
            for name in file_list:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i)+ "/")
    sA = g.StepAnalyzer(stepFiles,False,c)
    sA.DrawLengthPerAscent1(axes,"C"+str(num),num,c,marker = "*")

    axes[0].legend()
    num += 1

def analyze(folderName):
    condition = ["stair1_50","stair2_50","stair1_75","stair2_75","stair1_100","stair2_100"]#["stair1_60","stair1_85","stair1_100","stair2_30","stair2_60","stair2_85","stair2_100"]#["stair1_60","stair2_60","stair1_85","stair2_85"]
    #condition = ["stair1_50","stair2_50","stair1_75","stair2_75","stair1_100","stair2_100","stair1_half_50","stair1_half_75","stair1_half_100","stair2_half_50","stair2_half_75","stair2_half_100"]#,"stair2_50","stair1_75","stair2_75","stair1_100","stair2_100"]
    cDatas = dict()
    for c in condition:
        print(c)
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0,10):
            stepFiles.append(folderName + "박승준" + "/" + c + "/" + str(i)+ "/")
        cDatas[c] = g.StepAnalyzer(stepFiles,False)
    #plt.close()
    # print("compare(stair1_100,stair2_100)")
    # Compare2Result(cDatas["stair1_100"].avgDicts,cDatas["stair2_100"].avgDicts)
    # print("compare(stair1_85,stair2_85)")
    # Compare2Result(cDatas["stair1_85"].avgDicts,cDatas["stair2_85"].avgDicts)
    # print("compare(stair1_60,stair2_60)")
    # Compare2Result(cDatas["stair1_60"].avgDicts,cDatas["stair2_60"].avgDicts)
    comparePair = ["stair2_100","stair2_100"]
    # print("-----------------------compare(stair1_60, stair1_85)-----------------------------")
    # Compare2Result(cDatas["stair1_60"].avgDicts, cDatas["stair1_85"].avgDicts)
    # print("-----------------------compare(stair2_60, stair2_85)-----------------------------")
    # Compare2Result(cDatas["stair2_60"].avgDicts, cDatas["stair2_85"].avgDicts)
    print(comparePair)
    #Compare2Result(cDatas[comparePair[0]].avgDicts, cDatas[comparePair[1]].avgDicts)

    f, axes = plt.subplots(2, 1)
    for i in range(0,10):
        axes[0].set_title(comparePair[0])
        cDatas[comparePair[0]].data[i].DrawPosAndVelGraph(axes,color="C0")
        #cDatas[comparePair[0]].data[i].DrawPosAndVelGraph_z(axes)
        #cDatas[comparePair[0]].data[i].DrawPosAndVelGraph(axes,color="C1")
        #cDatas[comparePair[0]].data[i].DrawPosAndVelGraph(axes,color= "C0",label=comparePair[0],startIndex= cDatas[comparePair[0]].data[i].steps[0].validStart)
        #cDatas[comparePair[1]].data[i].DrawPosAndVelGraph(axes,color= "C1",label=comparePair[1],startIndex= cDatas[comparePair[1]].data[i].steps[0].validStart)

        #f, axes = plt.subplots(2, 2, sharey=True, sharex=True)
        # cDatas[comparePair[0]].data[i].DrawSectionPosAndVelGraph(axes,0,1," ("+comparePair[0]+")")
        # cDatas[comparePair[1]].data[i].DrawSectionPosAndVelGraph(axes,0, 1, " ("+comparePair[1]+")")
        #cDatas[comparePair[1]].data[i].DrawSectionPosAndVelGraph(axes,0, 1, " ("+comparePair[1]+")")
        #cDatas[comparePair[0]].data[i].DrawSectionPosAndVelGraph(axes,1,2," ("+comparePair[0]+")")
        # cDatas[comparePair[1]].data[i].DrawSectionPosAndVelGraph(axes,1, 2, " ("+comparePair[1]+")")
        #cDatas[comparePair[2]].data[i].DrawSectionPosAndVelGraph(axes,1, 2, " ("+comparePair[2]+")")
    plt.show()
    #for i in range(0, 10):
    #    cDatas["stair2_30"].data[i].DrawPosAndVelGraph(axes)

def ConvertRecordedData(inPath):
    condition = ["stair1_50","stair1_75","stair1_100"]
    for c in condition:
        for i in range(0,10):
            curPath = inPath + c + "/" + str(i) + "/"
            g.RecordedData(curPath).writeToTxt(curPath)



def test():
    for i in range(0,10):
        f, axes = plt.subplots(2, 1, sharey=True, sharex=True)
        plt.title(str(i))
        g.RecordedData(folder + "user3/이준혁/stair2_75/"+str(i)+"/", 2).DrawPosAndVelGraph(axes)
        plt.show()
#C:\Users\user\Desktop\Unity\VRStair\footdata\experiment\stair2_100_real

#f, axes = plt.subplots(2, 1, sharey=True, sharex=True)

#g.RecordedData(folder + "ex3_test/stiar2_Ours/1/", 1).DrawPosGraph(additionalLabel="(ours)")
#g.RecordedData(folder + "ex3_test/stiar2_Seo/1/", 1).DrawPosGraph(additionalLabel="(seo)")
#g.RecordedData(folder + "experiment/stair2_100_real/", 2).DrawPosGraph(additionalLabel="(real)",transX=25)
#g.RecordedData(folder + "ex3_test/stiar2_Ours/1/", 1).DrawPosAndVelGraph(axes,additionalLabel="(ours)")
#g.RecordedData(folder + "experiment/stair2_100_real/", 2).DrawPosAndVelGraph(axes,additionalLabel="(real)")
#g.RecordedData(folder + "experiment/stair2_100_ours/", 2).DrawPosAndVelGraph(axes,additionalLabel="(ours)")
#g.RecordedData(folder + "experiment/stair2_100_nagao/", 2).DrawPosAndVelGraph(axes,additionalLabel="(nagao)")
#g.RecordedData(folder + "experiment/stair2_100_seo/", 2).DrawPosAndVelGraph(axes,additionalLabel="(seo)")

#plt.show()

#lengthCompare1(folder+"user3/")
#reader2(folder+"user/")
#analyze(folder+"user3/")
reader(folder+"user3/")
#MakeHeadDataFrame(folder+"user3/")
#WriteDataFrame(folder+"user3/")
#MakeAllTrajectoryDataFrame(folder+"user3/","real")
#MakeAllTrajectoryDataFrame(folder+"ex3/our/","our")
#WriteDataFrame(folder+"user3/")
#reader1(folder)
# f, axes = plt.subplots(2, 1, sharey=True, sharex=True)
#g.RecordedData(folder+"user3/서민영/stair2_100/0/",2).DrawPosAndVelGraph(axes)
#plt.show()


