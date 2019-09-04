import numpy as np
import math

np.random.seed(42)


def RGBtoXYZ(color):
    comp = np.where(color > 0.04045, 1, 0)
    newColor = comp * np.power((color + 0.055)/1.055, 2.4) + (1-comp) * color/12.92
    newColor *= 100

    convMat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
        ])

    XYZ = np.einsum('kj,ij->ik', convMat, newColor) # Possible rounding of result

    return XYZ

def XYZtoLAB(XYZ):
    newXYZ = XYZ / np.array([95.047, 100.0, 108.883])

    comp = np.where(newXYZ > 0.008856, 1, 0)
    newXYZ = comp * np.power(newXYZ, 1/3.0) + (1-comp) * (7.787 * newXYZ + 16/116.0)

    convMat = np.array([
        [0, 116, 0],
        [500, -500, 0],
        [0, 200, -200]
    ])

    Lab = np.einsum('kj,ij->ik', convMat, newXYZ) + np.array([-16, 0, 0])

    return Lab
    
#--------------------------------------#

arr = np.random.rand(3,3)

print("Values :")
print(np.around(arr, 2))
print("XYZ :")
print(RGBtoXYZ(arr))
print("Lab :")
print(XYZtoLAB(RGBtoXYZ(arr)))