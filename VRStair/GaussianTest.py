import math
import matplotlib.pyplot as plt
import numpy as np

max_velocity = 1
h = 0.24


def erf(x):
    output = 2/math.pi
    t2 = x - x**3/ 3 + x**5/10 - x**7/42 + x**9/216
    return output * t2


def f(z, std):
    output = 1 / (std * math.pow(2 * math.pi, 1/2))
    output = output * math.pow(math.e, (-1/2) * math.pow(z / std, 2))
    return output

def normal_distribution(x, std):
    return 1 / (std * math.sqrt(2*math.pi)) * math.pow(math.e, -math.pow(x/std,2)/2) * h

def f_cdf(x, std):
    out = 0
    range = np.arange(0, abs(x), width)
    result = 0
    for j in range:
        out += normal_distribution(j, std) * width
    if x > 0:
        result = out + 1/2
    else:
        result = 1/2 - out

    return result


def skewGaussian(x,location, scale, shape, std):
    x = (x - location) / scale
    output = 2 / scale * normal_distribution(x, std) * f_cdf(shape*x, std)
    return output


arr = []
width = 0.011111
#range = np.arange(-1, 1, width)

ans = 0
ans2 = 0
i = 0

startTh = 0.05
threshold = 0.05

xAxis = []

xAxis = np.arange(-3, 3, width)
skew_max_height = 0
norm_max_height = 0
location, scale, shape = (0, 1, 10)
skew_mode_index = 0
norm_mode_index = 0
for i in xAxis:
    std = 1
    if norm_max_height < normal_distribution(i, std):
        norm_max_height = normal_distribution(i, std)
        norm_mode_index = i

    if skew_max_height < skewGaussian(i, location, scale, shape, std):
        skew_max_height = skewGaussian(i, location, scale, shape, std)
        skew_mode_index = i

testArr = []
norm_std = norm_max_height / max_velocity
skew_std = skew_max_height / max_velocity
location = -1 * (skew_mode_index - norm_mode_index)

x1 = []
x2 = []
for i in xAxis:
    h1 = normal_distribution(i,norm_std)
    if h1 > threshold:
        arr.append(normal_distribution(i, norm_std))
        x1.append(i)
        ans += normal_distribution(i, norm_std) * width

    h2 = skewGaussian(i, location, scale, shape, skew_std)
    if h2 > threshold:
        x2.append(i)
        testArr.append(skewGaussian(i, location, scale, shape, skew_std))
        ans2 += skewGaussian(i, location, scale, shape, skew_std) * width



plt.plot(x1, arr)
plt.plot(x2, testArr)
plt.show()

print(ans)
print(ans2)