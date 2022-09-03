
import GraphMaker as g
import matplotlib.pyplot as plt

folder = "C:/Users/user/Desktop/Unity/VRStair/footdata/recodingTest/"
data =  g.RecordedData(folder)
data1 = g.RecordedData(folder + "/36/")
data2 = g.RecordedData(folder + "/8/")
data2.DrawGrahp(color = "C0")
data1.DrawGrahp(color = "C4")
#data.DrawGrahp(x = "Distance")
#data1.DrawGrahp()
#data2.DrawGrahp()

folder2 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
Stair1 = g.RecordedData(folder2,2)
#Stair1.DrawGrahp(color= "C3")
# folder3 = "C:/Users/user/Desktop/Unity/VRStair/footdata/s1/0/"
# Stair2 = g.RecordedData(folder3,2)
# Stair2.DrawGrahp()
#Stair1.DrawGrahp(x = "Distance")
#Stair2.DrawGrahp(x = "Distance")
plt.show()


#Stair2.DrawGrahp()
#plt.show()