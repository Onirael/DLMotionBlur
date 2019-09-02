import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import cv2 as cv
import os
import imageio

filetype = ".png"
workDirectory = "Test2"


for imageFile in os.listdir(workDirectory) :

    #Check if the file is in the current file type
    filename = os.fsdecode(imageFile)
    if not filename.endswith(filetype):
        continue

    if filetype == '.exr' :
        baseImage = cv.imread(workDirectory + '/' + imageFile, cv.IMREAD_ANYDEPTH)
    elif filetype == '.png' :
        baseImage = io.imread(workDirectory + '/' + imageFile)
    elif filetype == '.hdr' :
        baseImage = imageio.imread(workDirectory + '/' + imageFile + '.hdr')

    adjImage = baseImage[:,:,:3]

    fileOutName = workDirectory + '/' + filename

    print("Writing image ", fileOutName)

    if filetype == '.exr' :
        cv.imwrite(fileOutName, adjImage)
    elif filetype == '.png' :
        io.imsave(fileOutName, adjImage, check_contrast=False, plugin='imageio')