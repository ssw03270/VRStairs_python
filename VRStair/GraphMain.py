
import GraphMaker as g
import matplotlib.pyplot as plt
import os

folder = "C:/Users/Dobby/Documents/GitHub/VRStair/footdata/"
#data =  g.RecordedData(folder)

#data.DrawGrahp(x = "Distance")
#data1.DrawGrahp()
#data2.DrawGrahp()

#folder2 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
#real = g.RecordedData(folder2,2)


def reader(folderName):
    condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]
    for c in condition:
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0,10):
            for name in file_list:
                stepFiles.append(folderName + name + "/" + c + "/" + str(i)+ "/")
        print(c)
        g.StepAnalyzer(stepFiles,True)
        #plt.show()

def reader1(folderName):
    condition = ["stair1","stair2","stair1_60","stair2_60","stair1_85","stair2_85"]
    for c in condition:
        file_list = os.listdir(folderName)
        stepFiles = []
        for i in range(0,10):
            stepFiles.append(folderName + "홍성은" + "/" + c + "/" + str(i)+ "/")
        print(c)
        g.StepAnalyzer(stepFiles,False)
#reader1(folder+"user/")
reader(folder+"user/")
#g.RecordedData("C:/Users/Dobby/Documents/GitHub/VRStair/footdata/user/홍성은/stair2_60/9/",2).DrawGrahp()
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