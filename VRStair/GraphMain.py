
import GraphMaker as g
import matplotlib.pyplot as plt

folder = "C:/Users/Dobby/Documents/GitHub/VRStair/"
#data =  g.RecordedData(folder)

#data.DrawGrahp(x = "Distance")
#data1.DrawGrahp()
#data2.DrawGrahp()

#folder2 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
#real = g.RecordedData(folder2,2)


names = ["강경은","김경민","노성래","박주현","정승재","주창돈","한동현","서민영","이로운","이준혁","임재원","조장현"]

stair1 = []
for i in range(0,10):
    for n in names:
        stair1.append("foot_dataset/real_data/"+n+ "/stair1/" + str(i) + "/")


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

print("stair1")
g.StepAnalyzer(stair1)
print("stair2")
g.StepAnalyzer(stair2)
print("stair1_60")
g.StepAnalyzer(stair1_60)
print("stair2_60")
g.StepAnalyzer(stair2_60)
print("stair1_85")
g.StepAnalyzer(stair1_85)
print("stair2_85")
g.StepAnalyzer(stair2_85)

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