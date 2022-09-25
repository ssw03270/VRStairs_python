
import GraphMaker as g
import matplotlib.pyplot as plt
import os
import csv

#folder = "C:/Users/Dobby/Documents/GitHub/VRStairs_python/VRStair/foot_dataset/"
folder = "C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
#folder = "C:/Users/user/Desktop/Unity/VRStair/footdata/"
#data =  g.RecordedData(folder)

#data.DrawGrahp(x = "Distance")
#data1.DrawGrahp()
#data2.DrawGrahp()
#f, axes = plt.subplots(2, 1)
#virtual = g.RecordedData(folder+"recodingTest/test/",1).DrawPosAndVelGraph(axes)
#virtual = g.RecordedData(folder+"recodingTest/test60/",1).DrawPosAndVelGraph(axes)
#folder2 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
#real = g.RecordedData(folder2,2)

def writeCSV(resultDict,condition,name = "avg"):
    with open(name+".csv",'w',encoding="UTF-8",newline="") as f:
        w = csv.writer(f)
        order = ["Head 1","Head 2", "First Foot", "Second Foot", "Last Foot"]
        w.writerow(["",""]+list(resultDict[condition[0]][0][0].keys()))
        for c in condition:
            curInfo = resultDict[c]
            w.writerow([c])
            i = 0
            for o in order:
                w.writerow([o])
                avg = list(curInfo[0][i].values())
                sd = list(curInfo[1][i].values())
                w.writerow(["","avg"]+avg)
                w.writerow(["","SD"]+sd)
                i += 1

# condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]
def reader(folderName):
    result = dict()
    condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]
    for c in condition:
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0,10):
            for name in file_list:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i)+ "/")
        print(c)
        result[c] = g.StepAnalyzer(stepFiles,False,c).GetResultList()
    print("-----------------------compare(stair1_60, stair1_85)-----------------------------")
    Compare2Result(result["stair1_60"][0], result["stair1_85"][0])
    print("-----------------------compare(stair2_60, stair2_85)-----------------------------")
    Compare2Result(result["stair2_60"][0], result["stair2_85"][0])
    #Compare2Result(result["stair1_60"][0],result["stair2_60"][0])
    #Compare2Result(result["stair1_85"][0], result["stair2_85"][0])
    writeCSV(result,condition,"old")
        #plt.show()

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
    order = ["Head 1", "Head 2", "First Foot", "Second Foot", "Last Foot"]
    for i in range(len(avgDict1)):
        print("--------------------",order[i], "---------------------------------")
        print("< diff >")
        for k in avgDict1[i].keys():
            print(k ,":" ,avgDict1[i][k]-avgDict2[i][k],end=",")
        print("\n< 2/1 > : ")
        for k in avgDict1[i].keys():
            print(k ,":" ,avgDict2[i][k]/avgDict1[i][k],end=",")
        print()


def analyze(folderName):
    condition = ["stair1_60","stair1_85","stair1_100","stair2_30","stair2_60","stair2_85","stair2_100"]#["stair1_60","stair1_85","stair1_100","stair2_30","stair2_60","stair2_85","stair2_100"]#["stair1_60","stair2_60","stair1_85","stair2_85"]
    #condition = ["stair1_60","stair2_60","stair1_85","stair2_85"]
    cDatas = dict()
    for c in condition:
        print(c)
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0,10):
            #stepFiles.append(folderName + "서승원" + "/" + c + "/" + str(i)+ "/")
            stepFiles.append(folderName + "서승원" + "/" + c + "/" + str(i) + "/")
        cDatas[c] = g.StepAnalyzer(stepFiles,False)
    #plt.close()
    # print("compare(stair1_100,stair2_100)")
    # Compare2Result(cDatas["stair1_100"].avgDicts,cDatas["stair2_100"].avgDicts)
    # print("compare(stair1_85,stair2_85)")
    # Compare2Result(cDatas["stair1_85"].avgDicts,cDatas["stair2_85"].avgDicts)
    # print("compare(stair1_60,stair2_60)")
    # Compare2Result(cDatas["stair1_60"].avgDicts,cDatas["stair2_60"].avgDicts)
    comparePair = ["stair1_100","stair2_30"]
    # print("-----------------------compare(stair1_60, stair1_85)-----------------------------")
    # Compare2Result(cDatas["stair1_60"].avgDicts, cDatas["stair1_85"].avgDicts)
    # print("-----------------------compare(stair2_60, stair2_85)-----------------------------")
    # Compare2Result(cDatas["stair2_60"].avgDicts, cDatas["stair2_85"].avgDicts)
    print(comparePair)
    Compare2Result(cDatas[comparePair[0]].avgDicts, cDatas[comparePair[1]].avgDicts)
    f, axes = plt.subplots(2, 1, sharey=True, sharex=True)
    for i in range(0,10):
        #cDatas[comparePair[0]].data[i].DrawPosAndVelGraph(axes,color="C0")
        #cDatas[comparePair[1]].data[i].DrawPosAndVelGraph(axes,color="C1")
        cDatas[comparePair[0]].data[i].DrawPosAndVelGraph(axes,color= "C0",label=comparePair[0],startIndex= cDatas[comparePair[0]].data[i].steps[0].validStart)
        cDatas[comparePair[1]].data[i].DrawPosAndVelGraph(axes,color= "C1",label=comparePair[1],startIndex= cDatas[comparePair[1]].data[i].steps[0].validStart)

        #f, axes = plt.subplots(2, 2, sharey=True, sharex=True)
        # cDatas[comparePair[0]].data[i].DrawSectionPosAndVelGraph(axes,0,1," ("+comparePair[0]+")")
        # cDatas[comparePair[1]].data[i].DrawSectionPosAndVelGraph(axes,0, 1, " ("+comparePair[1]+")")
        # #cDatas[comparePair[2]].data[i].DrawSectionPosAndVelGraph(axes,0, 1, " ("+comparePair[2]+")")
        # cDatas[comparePair[0]].data[i].DrawSectionPosAndVelGraph(axes,1,2," ("+comparePair[0]+")")
        # cDatas[comparePair[1]].data[i].DrawSectionPosAndVelGraph(axes,1, 2, " ("+comparePair[1]+")")
        #cDatas[comparePair[2]].data[i].DrawSectionPosAndVelGraph(axes,1, 2, " ("+comparePair[2]+")")
    plt.show()
    #for i in range(0, 10):
    #    cDatas["stair2_30"].data[i].DrawPosAndVelGraph(axes)



#reader2(folder+"user/")
#analyze(folder)
reader(folder+"user/")
#reader1(folder)
#g.RecordedData("C:/Users/Dobby/Documents/GitHub/VRStair/footdata/서승원/stair1_60/2/",2).DrawPosAndVelGraph()
#plt.show()
'''
file_list = os.listdir(folder + "user/")
print(file_list)
names = ["김민정","황준태"]

stair1 = []
for i in range(0,10):
    for n in names:
        stair1.append("foot_dataset/real_data/"+n+ "/stair1/" + str(i) + "/")
print("stair1")
g.StepAnalyzer(stair1,True)


stair1_60 = []
for i in range(0,10):
    for n in names:
        stair1_60.append("foot_dataset/real_data/"+n+ "/stair1_60/" + str(i) + "/")

stair1_85 = []
for i in range(0,10):
    for n in names:
        stair1_85.append("foot_dataset/real_data/"+n+ "/stair1_85/" + str(i) + "/")

stair2 = []
for i in range(0,10):
    for n in names:
        stair2.append("foot_dataset/real_data/"+n+ "/stair2/" + str(i) + "/")

stair2_60 = []
for i in range(0,10):
    for n in names:
        stair2_60.append("foot_dataset/real_data/"+n+ "/stair2_60/" + str(i) + "/")

stair2_85 = []
for i in range(0,10):
    for n in names:
        stair2_85.append("foot_dataset/real_data/"+n+ "/stair2_85/" + str(i) + "/")
'''

# print("stair2")
# g.StepAnalyzer(stair2)
# print("stair1_60")
# g.StepAnalyzer(stair1_60)
# print("stair2_60")
# g.StepAnalyzer(stair2_60)
# print("stair1_85")
# g.StepAnalyzer(stair1_85)
# print("stair2_85")
# g.StepAnalyzer(stair2_85)

#g.StepAnalyzer(files1)



#Stair1.DrawGrahp(color= "C3")
# folder3 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
# Stair2 = g.RecordedData(folder3,2)
# Stair2.DrawGrahp()
#Stair1.DrawGrahp(x = "Distance")
#Stair2.DrawGrahp(x = "Distance")


plt.show()


#Stair2.DrawGrahp()
#plt.show()