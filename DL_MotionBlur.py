from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from time import perf_counter_ns
import os, math, random, imageio, pickle
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

dataShape = 201 # Convolution K size

# Training
trainModel = True
trainFromCheckpoint = True
batchSize = 200
trainEpochs = 15
stride = 99
learningRate = 0.001
saveFiles = True

shuffleSeed = 36

# Debug & Visualization
lossGraph = True
testRender = False
debugSample = False
randomSample = False
sample = 420

# File handling
digitFormat = 4
setCount = 20
startFrame = 228
endFrame = 999
resourcesFolder = "D:/Bachelor_resources/"
workDirectory = resourcesFolder  + 'Capture1_Sorted/'
filePrefix = 'Capture1_'

# Model output
weightsFileName = resourcesFolder + "3Depth_K201_Weights.h5"
graphDataFileName = resourcesFolder + "3Depth_K201_GraphData.dat"

#------------------------TF session-------------------------#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#-----------------------Keras Callback----------------------#

trainCheckpoint = ModelCheckpoint(weightsFileName, verbose=0, save_weights_only=True)

#-----------------------Keras Sequence----------------------#

class SampleSequence(tf.keras.utils.Sequence) :
  def __init__(self, batch_size, frames, frameShape, sampleMaps, stride=0) :
    global dataShape

    self.frameShape = frameShape
    self.sampleSize = (dataShape - 1)//2                                                            # "Padding" of the convolution kernel
    self.batch_size = batch_size
    self.batchPerFrame = self.frameShape[0] * self.frameShape[1] // (batch_size * stride)           # Number of batches per input frames
    self.batchAmount = len(frames) * self.batchPerFrame                                             # Total number of batches
    self.frames = frames                                                                            # List of input frame numbers
    self.stride = stride                                                                            # Pixels to skip when reading file
    self.batchArray = np.arange(batch_size)
    self.sampleMaps = sampleMaps

  def __len__(self) :
    return self.batchAmount
  
  def __getitem__(self, idx) :
    global dataShape
    global filePrefix
    global digitFormat

    frameID = idx//self.batchPerFrame                                               # Gets the ID of the current frame
    frame = self.frames[frameID]                                                    # Gets the input frame number
    frameBatch = idx - frameID * self.batchPerFrame                                 # Gets the batch number for the current frame

    # Import frames
    sceneColor = PadImage(imageio.imread(workDirectory + 'SceneColor/' + filePrefix + 'SceneColor_' + GetFrameString(frame, digitFormat) + '.png')[:,:,:3]/255.0, self.sampleSize).astype('float16')
    sceneDepth0 = PadImage(imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(frame, digitFormat) + '.hdr')[:,:,:1]/3000.0, self.sampleSize).astype('float16')
    sceneDepth1 = PadImage(imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(frame - 1, digitFormat) + '.hdr')[:,:,:1]/3000.0, self.sampleSize).astype('float16')
    sceneDepth2 = PadImage(imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(frame - 2, digitFormat) + '.hdr')[:,:,:1]/3000.0, self.sampleSize).astype('float16')
    finalImage = imageio.imread(workDirectory + 'FinalImage/' + filePrefix + 'FinalImage_' + GetFrameString(frame, digitFormat) + '.png')[:,:,:3].astype('float16')

    # Batch arrays
    batch_SceneColor = np.zeros((self.batch_size, dataShape, dataShape, 3))
    batch_SceneDepth0 = np.zeros((self.batch_size, dataShape, dataShape, 1))
    batch_SceneDepth1 = np.zeros((self.batch_size, dataShape, dataShape, 1))
    batch_SceneDepth2 = np.zeros((self.batch_size, dataShape, dataShape, 1))
    batch_FinalImage = np.zeros((self.batch_size, 3))

    for element in range(self.batch_size) :
      i = (element + frameBatch * self.batch_size) * self.stride                       # Gets the pixel ID for the current frame
      samplePixel = self.sampleMaps[frameID, i%self.frameShape[0], i//self.frameShape[1]]
      pixel = (samplePixel[0] + self.sampleSize, samplePixel[1] + self.sampleSize)     # Gets the pixel coordinates

      # Array assignment
      batch_SceneColor[element] = SampleImage(sceneColor, pixel, self.sampleSize)
      batch_SceneDepth0[element] = SampleImage(sceneDepth0, pixel, self.sampleSize)
      batch_SceneDepth1[element] = SampleImage(sceneDepth1, pixel, self.sampleSize)
      batch_SceneDepth2[element] = SampleImage(sceneDepth2, pixel, self.sampleSize)
      batch_FinalImage[element] = finalImage[samplePixel[0], samplePixel[1]]
        
    return ({'input_0':batch_SceneColor, 'input_1':batch_SceneDepth0, 'input_2':batch_SceneDepth1, 'input_3':batch_SceneDepth2}, batch_FinalImage)

#-------------------------Functions-------------------------#

def shuffle_along_axis(a, axis): # Function courtesy of Divakar (https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis/5044364#5044364)
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def GetSampleMaps(frameShape, frames, seed) :
  indexMap = np.zeros((frameShape[0], frameShape[1], 2))
  indexMap[:,:,0] = np.reshape(np.tile(np.array([np.arange(frameShape[0])]), frameShape[1]), (frameShape[0], frameShape[1]))
  indexMap[:,:,1] = np.transpose(np.reshape(np.tile(np.array([np.arange(frameShape[1])]), frameShape[0]), (frameShape[1], frameShape[0])))
  sampleMaps = np.zeros((len(frames), frameShape[0], frameShape[1], 2))

  frameCount = len(frames)
  for i in range(frameCount) :
    np.random.seed(seed + i)
    sampleMap = shuffle_along_axis(shuffle_along_axis(indexMap, axis=0), axis=1)
    sampleMaps[i] = sampleMap
  
  np.random.seed(seed)

  return sampleMaps.astype('uint16')

def GetFrameString(frameNumber, digitFormat) : # Returns a string of the frame number with the correct amount of digits
  if math.log(frameNumber, 10) > digitFormat :
    raise ValueError("Digit format is too small for the frame number, {} for frame number {}".format(digitFormat, frameNumber))

  frameString = str(frameNumber)
  if (len(frameString) < digitFormat) :
    frameString = (digitFormat - len(frameString)) * "0" + frameString

  return frameString

def SampleImage(image, samplePixel, sampleSize) :
  shape = image.shape
  sample = np.array([])

  if (max(shape) < 2*sampleSize + 1) :
        print("invalid image or sample size ")
        return sample

  sample = image[samplePixel[0] - sampleSize:samplePixel[0] + sampleSize + 1, samplePixel[1] - sampleSize:samplePixel[1] + sampleSize + 1]

  return sample

def PadImage(image, sampleSize) : # Returns the image with a sampleSize large padding of zeros
  paddedImage = np.zeros((image.shape[0] + 2 * sampleSize, image.shape[1] + 2 * sampleSize, image.shape[2]))
  paddedImage[sampleSize:image.shape[0] + sampleSize, sampleSize:image.shape[1] + sampleSize] = image

  return paddedImage

def MakeRenderGenerator(sceneColor, sceneDepth0, sceneDepth1, sceneDepth2, frameShape, verbose=True) :

  for row in range(frameShape[0]) :
    if verbose:
      print("Rendering... ({:.2f}%)".format(row/frameShape[0] * 100), end="\r")

    curRow = \
    {
      'input_0' : np.zeros((frameShape[1], dataShape, dataShape, 3)),
      'input_1' : np.zeros((frameShape[1], dataShape, dataShape, 1)),
      'input_2' : np.zeros((frameShape[1], dataShape, dataShape, 1)),
      'input_3' : np.zeros((frameShape[1], dataShape, dataShape, 1)),
    }

    for column in range(frameShape[1]) :
      curRow['input_0'][column] = sceneColor[row:dataShape + row, column:dataShape + column]
      curRow['input_1'][column] = sceneDepth0[row:dataShape + row, column:dataShape + column]
      curRow['input_2'][column] = sceneDepth1[row:dataShape + row, column:dataShape + column]
      curRow['input_3'][column] = sceneDepth2[row:dataShape + row, column:dataShape + column]
    
    yield curRow

def ApplyKernel(image, flatKernel) : # Applies convolution kernel to same shaped image
  global dataShape
  kernel = tf.reshape(flatKernel, [tf.shape(flatKernel)[0], dataShape, dataShape])

  return tf.einsum('hij,hijk->hk', kernel, image)

def Loss(y_true, y_pred) : # Basic RGB color distance
  delta = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
  return delta

#-----------------------File handling-----------------------#

np.random.seed(shuffleSeed)
setDescription = np.random.randint(startFrame, endFrame, setCount) # Contains a random sample of frames to use as a data set
frameShape = imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(setDescription[0], digitFormat) + '.hdr').shape # Test image for shape

examplesCount = setCount * frameShape[0] * frameShape[1] /stride
examplesDisplayCount = examplesCount/1000000
print("\nTotal training examples : {:.2f} Million".format(examplesDisplayCount))
trainSetFraction = 1
crossValidSetFraction = 0.2
testSetFraction = 0.2

#Create generators
trainGenerator = SampleSequence(batchSize, setDescription, frameShape, GetSampleMaps(frameShape, setDescription, shuffleSeed), stride=stride//trainSetFraction)
crossValidGenerator = SampleSequence(batchSize, setDescription, frameShape, GetSampleMaps(frameShape, setDescription, shuffleSeed + 10), stride=int(stride//crossValidSetFraction))
testGenerator = SampleSequence(batchSize, setDescription, frameShape, GetSampleMaps(frameShape, setDescription, shuffleSeed + 20), stride=int(stride//testSetFraction))

print("\nTraining set size : {:.2f} Million".format(examplesDisplayCount * trainSetFraction))
print("Cross validation set size : {:.2f} Million".format(examplesDisplayCount * crossValidSetFraction))
print("Test set size : {:.2f} Million".format(examplesDisplayCount * testSetFraction))
print()


#-------------------------Debug-----------------------------#

if (debugSample) :
  dataSampleMaps = GetSampleMaps(frameShape, setDescription, shuffleSeed)
  sampleGenerator = SampleSequence(batchSize, setDescription, frameShape, dataSampleMaps, stride=1)

  dataBatch = sampleGenerator.__getitem__(0)
  dataExample = dataBatch[0]['input_0'][0]
  frameShape = dataExample.shape

  batchPerFrame = (frameShape[0] * frameShape[1])//(batchSize * stride)
  if randomSample :
    testFrame = random.randint(0, len(setDescription))
    testBatch = random.randint(0, batchPerFrame)
    testElement = random.randint(0, batchSize)
  else :
    testFrame = sample
    testBatch = random.randint(0, batchPerFrame)
    testElement = random.randint(0, batchSize)

  plotTitle = "Frame {} sample {}".format(testFrame, testBatch * batchSize + testElement)

  fig = plt.figure(figsize=(8,8))
  fig.suptitle(plotTitle, fontsize=16)

  example = sampleGenerator.__getitem__(testFrame * batchPerFrame + testBatch)

  fig.add_subplot(2, 2, 1)
  plt.imshow(example[0]['input_0'][testElement])
  fig.add_subplot(2, 2, 2)
  plt.imshow(example[0]['input_1'][testElement,:,:,0], cmap='gray')
  fig.add_subplot(2, 2, 3)
  plt.imshow(example[0]['input_2'][testElement,:,:,0], cmap='gray')
  fig.add_subplot(2, 2, 4)
  plt.imshow(example[1][testElement, np.newaxis, np.newaxis]/255.0)


  print("Max depth : ", np.amax(example[0]['input_1'][testElement]))
  plt.show()
  quit()

#---------------------TensorFlow model----------------------#

input0 = tf.keras.Input(shape=(dataShape, dataShape, 3), name='input_0') #Scene color
input1 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_1') #Depth 0
input2 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_2') #Depth -1
input3 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_3') #Depth -2

#-Definition---------------------#

#Input1
x = tf.keras.layers.MaxPooling2D(2,2)(input1)
x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(4,4)(x)
x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)
x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.Model(inputs=input1, outputs=x)

#Input2
y = tf.keras.layers.MaxPooling2D(2,2)(input2)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.MaxPooling2D(4,4)(y)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.MaxPooling2D(2,2)(y)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.Flatten()(y)
y = tf.keras.Model(inputs=input2, outputs=y)

#Input3
z = tf.keras.layers.MaxPooling2D(2,2)(input3)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.MaxPooling2D(4,4)(z)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.MaxPooling2D(2,2)(z)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.Flatten()(z)
z = tf.keras.Model(inputs=input3, outputs=z)

#Combine inputs
combined = tf.keras.layers.concatenate([x.output, y.output, z.output])

#Common network
n = tf.keras.layers.Dense(128, activation='relu')(combined)
n = tf.keras.layers.Dense(dataShape**2, activation='linear')(n)
n = tf.keras.layers.ReLU()(n)
n = tf.keras.layers.Lambda(lambda l: ApplyKernel(input0, l))(n)

#Model
model = tf.keras.Model(inputs=[input0, x.input, y.input, z.input], outputs=n)

#--------------------------------#

model.compile(loss=Loss, 
  optimizer=RMSprop(lr=learningRate))

model.summary()

if trainModel :
  if trainFromCheckpoint :
    model.load_weights(weightsFileName)

  training = model.fit_generator(
    trainGenerator,
    validation_data=crossValidGenerator,
    epochs=trainEpochs,
    callbacks=[trainCheckpoint]
  )

  # Get training and test loss histories
  training_loss = training.history['loss']
  test_loss = training.history['val_loss']

  epoch_count = range(1, len(training_loss) + 1) # Create count of the number of epochs

  with open(graphDataFileName, 'wb') as graphDataFile :
    pickle.dump((training_loss, test_loss, epoch_count, setCount * trainSetFraction), graphDataFile)

  if saveFiles :
    model.save_weights(weightsFileName)
    print("Saved weights to file")

else :
  model.load_weights(weightsFileName)
  with open(graphDataFileName, 'rb') as graphDataFile :
    training_loss, test_loss, epoch_count, trainingSetSize = pickle.load(graphDataFile)

if (lossGraph) :
  #-----------------Visualize loss history--------------------#
  plt.title("Training examples : {}".format(trainingSetSize))
  plt.plot(epoch_count, training_loss, 'r--')
  plt.plot(epoch_count, test_loss, 'b-')
  plt.legend(['Training Loss', 'Test Loss'])
  plt.xlabel('Epoch')
  plt.xlim(0, len(training_loss))
  plt.ylabel('Loss')
  plt.ylim(0, 70)
  plt.show()

#--------------------------Test Model--------------------------#

sampleGenerator = SampleSequence(batchSize, testSet)

dataExample = sampleGenerator.__getitem__(0)[0]['input_0'][0]
frameShape = dataExample.shape

batchPerFrame = (frameShape[0] * frameShape[1])//batchSize
if randomSample :
  testFrame = random.randint(0, len(testSet))
  testBatch = random.randint(0, batchPerFrame)
  testElement = random.randint(0, batchSize)
else :
  testFrame = sample
  testBatch = random.randint(0, batchPerFrame)
  testElement = random.randint(0, batchSize)

example = sampleGenerator.__getitem__(testFrame * batchPerFrame + testBatch)[testElement]

testPredict = model.predict(example[0], steps=math.ceil(testSetSize/batchSize))
testLoss = model.evaluate_generator(testGenerator)

# Display sample results for debugging purpose
print("Test color : ", testPredict)
print("Expected color : ", example[1])
print("Test loss : ", testLoss)

start = perf_counter_ns()
batchPredict = model.predict_generator(testGenerator)[testElement]
end = perf_counter_ns()

print("Time per image: {:.2f}ms ".format((end-start)/testSetSize/1000000.0))

#-------------------------Test Render--------------------------#

if testRender:
  fig = plt.figure(figsize=(8,8))
  
  render_0FinalImage = imageio.imread('D:/Bachelor_resources/Capture1/Capture1_FinalImage_0411.png')
  frameShape = render_0FinalImage.shape

  padSize = math.floor((dataShape - 1)/2)
  render_0SceneColor = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 3))
  render_0SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))
  render_1SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))
  render_2SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))

  render_0SceneColor[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneColor_0412.png')[:,:,:3]/255.0).astype('float16')
  render_0SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0412.hdr')[:,:,:1]/3000.0).astype('float16')
  render_1SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0411.hdr')[:,:,:1]/3000.0).astype('float16')
  render_2SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0410.hdr')[:,:,:1]/3000.0).astype('float16')
  
  renderGenerator = MakeRenderGenerator(render_0SceneColor, render_0SceneDepth, render_1SceneDepth, render_2SceneDepth, frameShape)
  renderedImage = model.predict_generator(renderGenerator, steps=frameShape[0])

  finalImage = np.reshape(renderedImage, frameShape)

  fig.add_subplot(2, 1, 1)
  plt.imshow(render_0FinalImage)

  fig.add_subplot(2, 1, 2)
  plt.imshow(finalImage)

  plt.show()

#--------------------------------------------------------------#