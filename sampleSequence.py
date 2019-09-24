import tensorflow as tf
import imageio, math
import numpy as np

def GetFrameString(frameNumber, digitFormat) : # Returns a string of the frame number with the correct amount of digits
  if frameNumber > 0 and math.log(frameNumber, 10) > digitFormat :
    raise ValueError("Digit format is too small for the frame number, {} for frame number {}".format(digitFormat, frameNumber))

  frameString = str(frameNumber)
  if (len(frameString) < digitFormat) :
    frameString = (digitFormat - len(frameString)) * "0" + frameString

  return frameString

def PadImage(image, sampleSize) : # Returns the image with a sampleSize large padding of zeros
  paddedImage = np.zeros((image.shape[0] + 2 * sampleSize, image.shape[1] + 2 * sampleSize, image.shape[2]))
  paddedImage[sampleSize:image.shape[0] + sampleSize, sampleSize:image.shape[1] + sampleSize] = image
  
  return paddedImage

def SampleImage(image, samplePixel, sampleSize) :
  shape = image.shape
  sample = np.array([])

  if (max(shape) < 2*sampleSize + 1) :
    print("invalid image or sample size ")
    return sample

  sample = image[samplePixel[0] - sampleSize:samplePixel[0] + sampleSize + 1, samplePixel[1] - sampleSize:samplePixel[1] + sampleSize + 1]

  return sample

#-----------------------Keras Sequence----------------------#
class SampleSequence(tf.keras.utils.Sequence) :
  def __init__(self, batch_size, frames, frameShape, sampleMaps, dataShape, filePrefix, digitFormat, workDirectory, stride=1) :
    self.frameShape = frameShape
    self.sampleSize = (dataShape - 1)//2                                                            # "Padding" of the convolution kernel
    self.batch_size = batch_size
    self.batchPerFrame = self.frameShape[0] * self.frameShape[1] // (batch_size * stride)           # Number of batches per input frames
    self.batchAmount = len(frames) * self.batchPerFrame                                             # Total number of batches
    self.frames = frames                                                                            # List of input frame numbers
    self.stride = stride                                                                            # Pixels to skip when reading file
    self.batchArray = np.arange(batch_size)
    self.sampleMaps = sampleMaps
    self.dataShape = dataShape
    self.workDirectory = workDirectory
    self.filePrefix = filePrefix
    self.digitFormat = digitFormat

  def __len__(self) :
    return self.batchAmount
  
  def __getitem__(self, idx) :

    frameID = idx//self.batchPerFrame                                               # Gets the ID of the current frame
    frame = self.frames[frameID]                                                    # Gets the input frame number
    frameBatch = idx - frameID * self.batchPerFrame                                 # Gets the batch number for the current frame

    # Import frames
    sceneColor = PadImage(imageio.imread(self.workDirectory + 'SceneColor/' + self.filePrefix + 'SceneColor_' + GetFrameString(frame, self.digitFormat) + '.png')[:,:,:3]/255.0, self.sampleSize).astype('float16')
    sceneDepth0 = PadImage(imageio.imread(self.workDirectory + 'SceneDepth/' + self.filePrefix + 'SceneDepth_' + GetFrameString(frame, self.digitFormat) + '.hdr')[:,:,:1]/3000.0, self.sampleSize).astype('float16')
    sceneDepth1 = PadImage(imageio.imread(self.workDirectory + 'SceneDepth/' + self.filePrefix + 'SceneDepth_' + GetFrameString(frame - 1, self.digitFormat) + '.hdr')[:,:,:1]/3000.0, self.sampleSize).astype('float16')
    sceneDepth2 = PadImage(imageio.imread(self.workDirectory + 'SceneDepth/' + self.filePrefix + 'SceneDepth_' + GetFrameString(frame - 2, self.digitFormat) + '.hdr')[:,:,:1]/3000.0, self.sampleSize).astype('float16')
    finalImage = imageio.imread(self.workDirectory + 'FinalImage/' + self.filePrefix + 'FinalImage_' + GetFrameString(frame, self.digitFormat) + '.png')[:,:,:3].astype('float16')

    # Batch arrays
    batch_SceneColor = np.zeros((self.batch_size, self.dataShape, self.dataShape, 3))
    batch_SceneDepth0 = np.zeros((self.batch_size, self.dataShape, self.dataShape, 1))
    batch_SceneDepth1 = np.zeros((self.batch_size, self.dataShape, self.dataShape, 1))
    batch_SceneDepth2 = np.zeros((self.batch_size, self.dataShape, self.dataShape, 1))
    batch_FinalImage = np.zeros((self.batch_size, 3))

    for element in range(self.batch_size) :
      i = (element + frameBatch * self.batch_size) * self.stride                       # Gets the pixel ID for the current frame
      samplePixel = self.sampleMaps[frameID, i%self.frameShape[0], \
        i//self.frameShape[0]]
      pixel = (samplePixel%self.frameShape[0] + self.sampleSize, \
        samplePixel//self.frameShape[0] + self.sampleSize)     # Gets the pixel coordinates

      # Array assignment
      batch_SceneColor[element] = SampleImage(sceneColor, pixel, self.sampleSize)
      batch_SceneDepth0[element] = SampleImage(sceneDepth0, pixel, self.sampleSize)
      batch_SceneDepth1[element] = SampleImage(sceneDepth1, pixel, self.sampleSize)
      batch_SceneDepth2[element] = SampleImage(sceneDepth2, pixel, self.sampleSize)
      batch_FinalImage[element] = finalImage[samplePixel%self.frameShape[0], \
                                              samplePixel//self.frameShape[0]]
        
    return ({'input_0':batch_SceneColor, 'input_1':batch_SceneDepth0, 'input_2':batch_SceneDepth1, 'input_3':batch_SceneDepth2}, batch_FinalImage)

class RenderSequence(tf.keras.utils.Sequence) :
    
    def __init__(self, sceneColor, sceneDepth0, sceneDepth1, sceneDepth2, frameShape, dataShape, rowSteps=4, verbose=True) :
        self.sceneColor = sceneColor
        self.sceneDepth0 = sceneDepth0
        self.sceneDepth1 = sceneDepth1
        self.sceneDepth2 = sceneDepth2
        self.frameShape = frameShape
        self.dataShape = dataShape
        self.rowSteps = rowSteps
        self.verbose = verbose
        self.batchSize = math.floor(frameShape[1]/rowSteps)

    def __len__(self) :
        return self.rowSteps * self.frameShape[0]

    def __getitem__(self, idx) :
        row = idx//self.rowSteps
        rowStep = idx%self.rowSteps 
        if self.verbose :
            print("Rendering... ({:.2f}%)".format(row/self.frameShape[0] * 100), end="\r")

        curBatch = \
        {
          'input_0' : np.zeros((self.batchSize, self.dataShape, self.dataShape, 3)),
          'input_1' : np.zeros((self.batchSize, self.dataShape, self.dataShape, 1)),
          'input_2' : np.zeros((self.batchSize, self.dataShape, self.dataShape, 1)),
          'input_3' : np.zeros((self.batchSize, self.dataShape, self.dataShape, 1)),
        }
        
        for batchColumn in range(self.batchSize) :
            column = rowStep * self.batchSize + batchColumn
            curBatch['input_0'][batchColumn] = self.sceneColor[row:self.dataShape + row, column:self.dataShape + column]
            curBatch['input_1'][batchColumn] = self.sceneDepth0[row:self.dataShape + row, column:self.dataShape + column]
            curBatch['input_2'][batchColumn] = self.sceneDepth1[row:self.dataShape + row, column:self.dataShape + column]
            curBatch['input_3'][batchColumn] = self.sceneDepth2[row:self.dataShape + row, column:self.dataShape + column]

        return curBatch