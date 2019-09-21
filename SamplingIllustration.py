import numpy as np
import tensorflow as tf
import imageio, math, random
from matplotlib import pyplot as plt

dataShape = 201 # Convolution K size
batchSize = 200
stride = 10000
exampleFrame = 839
shuffleSeed = 36

# File handling
digitFormat = 4
startFrame = 228
endFrame = 999
resourcesFolder = "D:/Bachelor_resources/"
workDirectory = resourcesFolder  + 'Capture1_Sorted/'
filePrefix = 'Capture1_'

def SampleImage(image, samplePixel, sampleSize) :
  shape = image.shape
  sample = np.array([])

  if (max(shape) < 2*sampleSize + 1) :
        print("invalid image or sample size ")
        return sample

  sample = image[samplePixel[0] - sampleSize:samplePixel[0] + sampleSize + 1, samplePixel[1] - sampleSize:samplePixel[1] + sampleSize + 1]

def shuffle_along_axis(a, axis): # Function courtesy of Divakar (https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis/5044364#5044364)
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def GetSampleMaps(frameShape, frames, seed, shuffle=True) :
  indexMap = np.zeros((frameShape[0], frameShape[1], 2))
  indexMap[:,:,1] = np.reshape(np.tile(np.array([np.arange(frameShape[1])]), frameShape[0]), (frameShape[0], frameShape[1]))
  indexMap[:,:,0] = np.transpose(np.reshape(np.tile(np.array([np.arange(frameShape[0])]), frameShape[1]), (frameShape[1], frameShape[0])))
  sampleMaps = np.zeros((len(frames), frameShape[0], frameShape[1], 2))

  frameCount = len(frames)
  for i in range(frameCount) :
    np.random.seed(seed + i)
    if shuffle :
      sampleMap = shuffle_along_axis(shuffle_along_axis(indexMap, axis=1), axis=0)
    else :
      sampleMap = indexMap

    sampleMaps[i] = sampleMap
  
  np.random.seed(seed)

  return sampleMaps.astype('uint16')

def PadImage(image, sampleSize) : # Returns the image with a sampleSize large padding of zeros
  paddedImage = np.zeros((image.shape[0] + 2 * sampleSize, image.shape[1] + 2 * sampleSize, image.shape[2]))
  paddedImage[sampleSize:image.shape[0] + sampleSize, sampleSize:image.shape[1] + sampleSize] = image

  return paddedImage

def GetFrameString(frameNumber, digitFormat) : # Returns a string of the frame number with the correct amount of digits
  if math.log(frameNumber, 10) > digitFormat :
    raise ValueError("Digit format is too small for the frame number, {} for frame number {}".format(digitFormat, frameNumber))

  frameString = str(frameNumber)
  if (len(frameString) < digitFormat) :
    frameString = (digitFormat - len(frameString)) * "0" + frameString

  return frameString

def HighlightSamples(image, samplePixels, stride, brightness) :
    global dataShape
    sampleSize = (dataShape - 1)//2
    padPixels = samplePixels + sampleSize
    pixelAmount = len(padPixels)

    i = 0
    for pixel in padPixels :
        image[int(pixel[0]) - sampleSize:int(pixel[0]) + sampleSize + 1, \
            int(pixel[1]) - sampleSize:int(pixel[1]) + sampleSize + 1] = i/pixelAmount
        i += 1

frameShape = (1080, 1920)

shuffledSMap = GetSampleMaps(frameShape, np.array([exampleFrame]), shuffleSeed)[0]
unshuffledSMap = GetSampleMaps(frameShape, np.array([exampleFrame]), shuffleSeed, shuffle=False)[0]

shuffledSamplesRender = \
    PadImage(imageio.imread(workDirectory + 'SceneColor/' + filePrefix + 'SceneColor_' + GetFrameString(exampleFrame, digitFormat) + '.png')[:,:,:3]/255.0, \
        (dataShape - 1)//2)

unshuffledSamplesRender = \
    np.copy(shuffledSamplesRender)

shuffledSPixels = np.zeros((frameShape[0] * frameShape[1] // stride, 2))
unshuffledSPixels = np.zeros((frameShape[0] * frameShape[1] // stride, 2))

for i in range(frameShape[0] * frameShape[1] // stride) :
    index = i * stride
    shuffledSPixels[i] = shuffledSMap[index%frameShape[0], index//frameShape[0]]
    unshuffledSPixels[i] = unshuffledSMap[index%frameShape[0], index//frameShape[0]]

HighlightSamples(shuffledSamplesRender, shuffledSPixels, stride, 0.01)
HighlightSamples(unshuffledSamplesRender, unshuffledSPixels, stride, 0.01)

fig = plt.figure(figsize=(8,8))

fig.add_subplot(2, 1, 1)
plt.imshow(shuffledSamplesRender)

fig.add_subplot(2, 1, 2)
plt.imshow(unshuffledSamplesRender)

plt.show()