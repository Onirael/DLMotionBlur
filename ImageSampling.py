import numpy as np
from matplotlib import pyplot as plt
from skimage import data, io
import cv2 as cv
import imageio, math, random, os

#-----------------------------User Input--------------------------------------#

sSize = 100
samples = 100
frameAmount = 999
frameOffset = 2
startFrame = 226 #Number of first frame to compute
frameGap = 0
pixelMargin = True
digitFormat = 4 #Number of digits in the frame number identifier
fileTypes = [".png", ".exr", ".hdr"]
workDirectory = 'D:/Bachelor_resources/'
inputFolder = 'Capture1'
outputDirectory = workDirectory + '/' + 'samples3_' + inputFolder
inputList = ['0FinalImage', '0SceneDepth', '1SceneDepth', '0SceneColor']

#-----------------------------------------------------------------------------#

def sampleImage(image, samplePixel, sampleSize) :
    shape = image.shape
    sample = np.array([])

    if (max(shape) < 2*sampleSize + 1) :
        print("invalid image or sample size ")
        return sample

    sample = image[samplePixel[0] - sampleSize - 1:samplePixel[0] + sampleSize, samplePixel[1] - sampleSize - 1:samplePixel[1] + sampleSize]

    return sample

def randomSample(image, sampleSize) :
    shape = image.shape
    samplePixel = (0,0)

    samplePixel = (random.randint(sampleSize + 1, shape[0] - (sampleSize + 1)), random.randint(sampleSize + 1, shape[1] - (sampleSize + 1)))

    return samplePixel

def colorImage(image, samplePixel, sampleSize) :

    image[samplePixel[0] - sampleSize - 1:samplePixel[0] + sampleSize, samplePixel[1] - sampleSize - 1:samplePixel[1] + sampleSize] = 255

    return image

#-----------------------------------------------------------------------------#

testImage = io.imread(workDirectory + inputFolder + '/' + os.listdir(workDirectory + inputFolder)[0])
iterations = int((1.0 * frameAmount)/(frameGap + 1) - startFrame)
exportDigitFormat = math.floor(math.log(samples * frameAmount, 10)) + 1
subFolder = ''

print("Reading frames {} to {}\n".format(startFrame, iterations + startFrame))

for iteration in range(startFrame, iterations + startFrame) :

    frame = iteration * (frameGap + 1) + frameOffset #Add 1 to the frame value to ignore frame 0
    print("Frame : ", frame)
    frameString = ""

    pixels = [] #Used to store all computed pixel values for debugging

    for i in range(samples) :
        validPixel = False
        while (not validPixel) :
            newPixel = randomSample(testImage, sSize) #Compute a random pixel in range
            if newPixel not in pixels :
                pixels.append(newPixel)
                validPixel = True 

    for input in inputList :
        #Set the input type and define the format
        inputType = input[1:]
        if inputType in ['FinalImage', 'SceneColor'] :
            fileType = '.png'
        elif inputType in ['SceneDepth'] :
            fileType = '.hdr'
        
        #Set the frame string for the file name
        if input[0] == '0' :
            frameString = str(frame)
        else  :
            frameString = str(frame-int(input[0]))

        if (len(frameString) < digitFormat) :
            frameString = (digitFormat - len(frameString)) * "0" + frameString
        
        filename = inputFolder + '_' + inputType + '_' + frameString + fileType

        valid = False
        for extension in fileTypes :
            if filename.endswith(extension) :
                valid = True
                fileType = extension
        if not valid :
            continue

        if fileType == '.exr' :
            baseImage = cv.imread(workDirectory + inputFolder + '/' + filename, cv.IMREAD_ANYDEPTH)
        elif fileType == '.png' :
            baseImage = io.imread(workDirectory + inputFolder + '/' + filename)
        elif fileType == '.hdr' :
            baseImage = imageio.imread(workDirectory + inputFolder + '/' + filename)
        
        if inputType != 'FinalImage' and pixelMargin : #Add pixel margin
            marginImage = np.zeros((baseImage.shape[0] + 2 * sSize, baseImage.shape[1] + 2 * sSize, baseImage.shape[2]))
            marginImage[sSize:baseImage.shape[0] + sSize , sSize:baseImage.shape[1] + sSize] = baseImage
            baseImage = marginImage

        if inputType == 'SceneDepth' :
            baseImage = baseImage[:,:,:1]
        else :
            baseImage = baseImage[:,:,:3]
            baseImage = baseImage.astype('uint8') 
        
        for i in range(samples) :
            pixel = pixels[i]
            if inputType == 'FinalImage' :
                imageSample = np.array([[baseImage[pixel[0], pixel[1]]]]) #Sample the single pixel of the final image
                testExport = sampleImage(baseImage, pixel, sSize)
            else :
                imageSample = sampleImage(baseImage, pixel, sSize) #Sample pixels of image

            outputID = str(i + (iteration - startFrame) * samples)
            if (len(outputID) < exportDigitFormat) :
                outputID = (exportDigitFormat - len(outputID)) * "0" + outputID

            if inputType in ['SceneDepth', 'SceneColor'] :
                subFolder = 'Input'
            if inputType == 'FinalImage' :
                subFolder = 'Output'

            fileOutName = outputDirectory + '/' + subFolder + '/' + inputFolder + '_' + input + '_' + outputID + fileType
            print("Writing sample to file", fileOutName)

            if fileType in ['.exr', '.hdr'] :
                cv.imwrite(fileOutName, imageSample)
            elif fileType == '.png' :
                try :
                    io.imsave(fileOutName, imageSample, check_contrast=False, plugin='imageio')
                except ValueError :
                    print("Error when saving")
                    print(imageSample.shape)
                    print(pixel)
                    quit()