import numpy as np
import math

np.random.seed(42)

def RGBtoLCh(color):
    alpha = 0.01 * np.amin(color)/np.amax(color)
    y = 1
    Q = math.exp(alpha * y)
    L = (Q * np.amax(color) + (1-Q) * np.amin(color))/2.0
    C = (Q * (np.abs(color[0]-color[1]) + np.abs(color[1]-color[2]) + np.abs(color[2]-color[0])))/3.0
    H = math.atan2(color[1]-color[2], color[0]-color[1])

    return np.array([L,C,H])


def RGBtoXYZ(color):
    convMat = np.array([
        [0.412453, 0.35758 , 0.180423],
        [0.212671, 0.71516, 0.072169],
        [0.019334, 0.119193,  0.950227]])
    return np.matmul(convMat, color)
    
#--------------------------------------#

arr = np.random.rand(20,3)

print(arr[0])
print(RGBtoXYZ(arr[0]) * 100)