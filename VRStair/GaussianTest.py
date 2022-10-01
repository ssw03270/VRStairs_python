import math
import matplotlib.pyplot as plt
import numpy as np

def linear_regression(x):
    w = 3.8329531198263145
    b = 0.0667026156828887
    return w * x + b

width = 0.011111


def erf(x):
    output = 2/math.pi
    t2 = x - x**3/ 3 + x**5/10 - x**7/42 + x**9/216
    return output * t2


def f(z, std):
    output = 1 / (std * math.pow(2 * math.pi, 1/2))
    output = output * math.pow(math.e, (-1/2) * math.pow(z / std, 2))
    return output

def normal_distribution(x, std, h, max_velocity):
    return 1 / (std * math.sqrt(2*math.pi)) * math.pow(math.e, -math.pow(x/std,2)/2) * h

def f_cdf(x, std, h, max_velocity):
    out = 0
    range = np.arange(0, abs(x), width)
    result = 0
    for j in range:
        out += normal_distribution(j, std, h, max_velocity) * width
    if x > 0:
        result = out + 1/2
    else:
        result = 1/2 - out

    return result


def skewGaussian(x, location, scale, shape, std, h, max_velocity):
    x = (x - location) / scale
    output = 2 / scale * normal_distribution(x, std, h, max_velocity) * f_cdf(shape*x, std, h, max_velocity)
    return output

def calcGaussian(push, foot_velocity, stair_height):
    start_velocity = foot_velocity
    max_velocity = linear_regression(start_velocity)
    h = stair_height

    arr = []
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
    location, scale, shape = (0, 1, -10)
    skew_mode_index = 0
    norm_mode_index = 0
    for i in xAxis:
        std = 1
        if norm_max_height < normal_distribution(i, std, h, max_velocity):
            norm_max_height = normal_distribution(i, std, h, max_velocity)
            norm_mode_index = i

        if skew_max_height < skewGaussian(i, location, scale, shape, std, h, max_velocity):
            skew_max_height = skewGaussian(i, location, scale, shape, std, h, max_velocity)

    norm_std = norm_max_height / max_velocity
    skew_std = skew_max_height / max_velocity

    for i in xAxis:
        std = 1
        if norm_max_height < normal_distribution(i, std, h, max_velocity):
            norm_max_height = normal_distribution(i, std, h, max_velocity)
            norm_mode_index = i

        if skew_max_height < skewGaussian(i, location, scale, shape, std, h, max_velocity):
            skew_mode_index = i

    testArr = []
    location = -1 * (skew_mode_index - norm_mode_index)

    x1 = []
    x2 = []

    first_x1 = -999
    first_x2 = -999
    for i in xAxis:
        h1 = normal_distribution(i,norm_std, h, max_velocity)
        if h1 > threshold:
            if first_x1 == -999:
                first_x1 = i
            arr.append(normal_distribution(i, norm_std, h, max_velocity))
            x1.append(push + i - first_x1)
            ans += normal_distribution(i, norm_std, h, max_velocity) * width

        h2 = skewGaussian(i, location, scale, shape, skew_std, h, max_velocity)
        if h2 > threshold:
            if first_x2 == -999:
                first_x2 = i
            x2.append(push + i - first_x2)
            testArr.append(skewGaussian(i, location, scale, shape, skew_std, h, max_velocity))
            ans2 += skewGaussian(i, location, scale, shape, skew_std, h, max_velocity) * width

    print("ans2", ans2)

    plt.plot(x1, arr)
    plt.plot(x2, testArr)
# plt.show()
#
# print(ans)
# print(ans2)