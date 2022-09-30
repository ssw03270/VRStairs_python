import math
import matplotlib.pyplot as plt
import numpy as np

max_velocity = 0.38
h = 0.48


def erf(x):
    output = 2/math.pi
    t2 = x - x**3/ 3 + x**5/10 - x**7/42 + x**9/216
    return output * t2


def f(z, std):
    output = 1 / (std * math.pow(2 * math.pi, 1/2))
    output = output * math.pow(math.e, (-1/2) * math.pow(z / std, 2))
    return output

def normal_distribution(x):
    return 1 / math.sqrt(2*math.pi) * math.pow(math.e, -math.pow(x,2)/2)

def f_cdf(x):
    out = 0
    range = np.arange(0, abs(x), width)
    result = 0
    for j in range:
        out += normal_distribution(j) * width
    if x > 0:
        result = out + 1/2
    else:
        result = 1/2 - out

    return result


def skewGaussian(x,alpha):
    #output = 2 * f(0,1) * ( 1/2*(1+erf(alpha * x/math.sqrt(2))))
    output = 2 * normal_distribution(x) * f_cdf(alpha*x)
    return output


arr = []
width = 0.011111
#range = np.arange(-1, 1, width)

ans = 0
i = 0

startTh = 0.05
threshold = 0.05

xAxis = []

xAxis = np.arange(-3, 3, width)
testArr = []
for i in xAxis:
    arr.append(normal_distribution(i))
    testArr.append(skewGaussian(i,2))

print(max(testArr))

plt.plot(xAxis, arr)
plt.plot(xAxis, testArr)
plt.show()





print(ans)