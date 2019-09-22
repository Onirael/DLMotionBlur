from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.keras.mixed_precision.experimental import LossScaleOptimizer
from time import perf_counter_ns
import os, math, random, imageio, pickle
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

dataShape = 193 # Convolution K size

# Training
trainModel = True
modelFromFile = False
trainFromCheckpoint = False
batchSize = 128
trainEpochs = 10
stride = 25
learningRate = 0.001
saveFiles = True

shuffleSeed = 36

# Debug & Visualization
lossGraph = False
testRender = True
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
modelName = "3Depth_K193"
weightsImport = resourcesFolder + modelName + "_Weights.h5"
weightsFileName = resourcesFolder + modelName + "_ReLU" + "_Weights.h5"
graphDataFileName = resourcesFolder + modelName + "_GraphData.dat"

#------------------------TF session-------------------------#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#-----------------------Keras Callback----------------------#

trainCheckpoint = ModelCheckpoint(weightsFileName, verbose=0, save_weights_only=True)

#-----------------------Keras Sequence----------------------#

class SampleSequence(tf.keras.utils.Sequence) :
  def __init__(self, batch_size, frames, frameShape, sampleMaps, stride=1) :
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
    sceneColor = PadImage(imageio.imread(workDirectory + 'SceneColor/' + filePrefix + 'SceneColor_' + GetFrameString(frame, digitFormat) + '.png')\
      [:,:,:3]/255.0, self.sampleSize).astype('uint8')
    sceneDepth0 = PadImage(imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(frame, digitFormat) + '.hdr')\
      [:,:,:1]/3000.0, self.sampleSize).astype('float16')
    sceneDepth1 = PadImage(imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(frame - 1, digitFormat) + '.hdr')\
      [:,:,:1]/3000.0, self.sampleSize).astype('float16')
    sceneDepth2 = PadImage(imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(frame - 2, digitFormat) + '.hdr')\
      [:,:,:1]/3000.0, self.sampleSize).astype('float16')
    finalImage = imageio.imread(workDirectory + 'FinalImage/' + filePrefix + 'FinalImage_' + GetFrameString(frame, digitFormat) + '.png')[:,:,:3].astype('uint8')

    # Batch arrays
    batch_SceneColor = np.zeros((self.batch_size, dataShape, dataShape, 3))
    batch_SceneDepth0 = np.zeros((self.batch_size, dataShape, dataShape, 1))
    batch_SceneDepth1 = np.zeros((self.batch_size, dataShape, dataShape, 1))
    batch_SceneDepth2 = np.zeros((self.batch_size, dataShape, dataShape, 1))
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

#-------------------------Functions-------------------------#

def GetSampleMaps(frameShape, frames, seed) :
  sampleMaps = np.zeros((len(frames), frameShape[0], frameShape[1]))
  indexMap = np.reshape(np.arange(frameShape[0]*frameShape[1]), (frameShape[0], frameShape[1]))

  frameCount = len(frames)
  for i in range(frameCount) :
    np.random.seed(seed + i)

    sampleMap = np.copy(indexMap)
    np.random.shuffle(sampleMap)
    sampleMaps[i] = sampleMap
  np.random.seed(seed)

  return sampleMaps.astype('uint32')

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

def MakeRenderGenerator(sceneColor, sceneDepth0, sceneDepth1, sceneDepth2, frameShape, rowSteps, verbose=True) :
  batch_size = frameShape[1]//rowSteps

  for row in range(frameShape[0]) :
    if verbose:
      print("Rendering... ({:.2f}%)".format(row/frameShape[0] * 100), end="\r")

    for step in range(rowSteps) :
      curBatch = \
      {
        'input_0' : np.zeros((batch_size, dataShape, dataShape, 3)),
        'input_1' : np.zeros((batch_size, dataShape, dataShape, 1)),
        'input_2' : np.zeros((batch_size, dataShape, dataShape, 1)),
        'input_3' : np.zeros((batch_size, dataShape, dataShape, 1)),
      }
      for batchElement in range(batch_size) :
        column = (step * batch_size + batchElement)
        curBatch['input_0'][batchElement] = sceneColor[row:dataShape + row, column:dataShape + column]
        curBatch['input_1'][batchElement] = sceneDepth0[row:dataShape + row, column:dataShape + column]
        curBatch['input_2'][batchElement] = sceneDepth1[row:dataShape + row, column:dataShape + column]
        curBatch['input_3'][batchElement] = sceneDepth2[row:dataShape + row, column:dataShape + column]
    
      yield curBatch

def ApplyKernel(image, flatKernel) : # Applies convolution kernel to same shaped image
  global dataShape
  kernel = tf.reshape(flatKernel, [tf.shape(flatKernel)[0], dataShape, dataShape])

  return tf.einsum('hij,hijk->hk', kernel, image)

def Loss(y_true, y_pred) : # Basic RGB color distance
  # y_pred = K.cast(y_pred, dtype='float32')
  delta = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
  return delta

def RenderLoss(y_true, y_pred) :
  delta = np.mean(np.absolute(y_true - y_pred), axis=2)
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
print("\nGenerating trainig data...")
trainGenerator = SampleSequence(batchSize, setDescription, frameShape, \
  GetSampleMaps(frameShape, setDescription, shuffleSeed), stride=stride//trainSetFraction)
crossValidGenerator = SampleSequence(batchSize, setDescription, frameShape, \
  GetSampleMaps(frameShape, setDescription, shuffleSeed + 10), stride=int(stride//crossValidSetFraction))
testGenerator = SampleSequence(batchSize, setDescription, frameShape, \
  GetSampleMaps(frameShape, setDescription, shuffleSeed + 20), stride=int(stride//testSetFraction))

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

K.set_floatx('float16')
K.set_epsilon(1e-8)

input0 = tf.keras.Input(shape=(dataShape, dataShape, 3), name='input_0', dtype='float16') #Scene color
input1 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_1', dtype='float16') #Depth 0
input2 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_2', dtype='float16') #Depth -1
input3 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_3', dtype='float16') #Depth -2

if modelFromFile :
  # load json and create model
  with open(resourcesFolder + modelName + '.json', 'r') as json_file :
    loaded_model_json = json_file.read()

  model = tf.keras.models.model_from_json(loaded_model_json)
  print("Loaded model from disk")

else :
  #-Definition---------------------#

  #Input1
  x0 = tf.keras.layers.MaxPooling2D((2,2), padding='same')(input1)
  x0 = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu')(x0)
  x0 = tf.keras.layers.ReLU()(x0)
  x1 = tf.keras.layers.MaxPooling2D((2,2), padding='same')(x0)
  x1 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(x1)
  x1 = tf.keras.layers.ReLU()(x1)
  x2 = tf.keras.layers.MaxPooling2D(2,2)(x1)
  x2 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x2)
  x2 = tf.keras.layers.ReLU()(x2)
  x = tf.keras.Model(inputs=input1, outputs=x2)

  #Input2
  y0 = tf.keras.layers.MaxPooling2D((2,2), padding='same')(input2)
  y0 = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu')(y0)
  y0 = tf.keras.layers.ReLU()(y0)
  y1 = tf.keras.layers.MaxPooling2D((2,2), padding='same')(y0)
  y1 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(y1)
  y1 = tf.keras.layers.ReLU()(y1)
  y2 = tf.keras.layers.MaxPooling2D(2,2)(y1)
  y2 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(y2)
  y2 = tf.keras.layers.ReLU()(y2)
  y = tf.keras.Model(inputs=input2, outputs=y2)

  #Input3
  z0 = tf.keras.layers.MaxPooling2D((2,2), padding='same')(input3)
  z0 = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu')(z0)
  z0 = tf.keras.layers.ReLU()(z0)
  z1 = tf.keras.layers.MaxPooling2D((2,2), padding='same')(z0)
  z1 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(z1)
  z1 = tf.keras.layers.ReLU()(z1)
  z2 = tf.keras.layers.MaxPooling2D(2,2)(z1)
  z2 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(z2)
  z2 = tf.keras.layers.ReLU()(z2)
  z = tf.keras.Model(inputs=input3, outputs=z2)


  #Combine inputs
  combined = tf.keras.layers.Add()([x.output, y.output, z.output])
  combined = tf.keras.layers.UpSampling2D((2,2))(combined)
  combined = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(combined)
  combined = tf.keras.layers.ReLU()(combined)

  combined = tf.keras.layers.concatenate([x1, y1, z1])
  combined = tf.keras.layers.UpSampling2D((2,2))(combined)
  combined = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu')(combined)
  combined = tf.keras.layers.ReLU()(combined)
  combined = tf.keras.layers.Cropping2D(cropping=((1,0), (1,0)))(combined)

  combined = tf.keras.layers.Add()([x0, y0, z0, combined])
  combined = tf.keras.layers.UpSampling2D((2,2))(combined)
  combined = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu')(combined)
  combined = tf.keras.layers.ReLU()(combined)
  combined = tf.keras.layers.Cropping2D(cropping=((1,0), (1,0)))(combined)
  combined = tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='relu')(combined)
  combined = tf.keras.layers.ReLU()(combined)


  #Common network
  n = tf.keras.layers.Lambda(lambda l: ApplyKernel(input0, l))(combined)

  #Model
  model = tf.keras.Model(inputs=[input0, x.input, y.input, z.input], outputs=n, name=modelName)

  #--------------------------------#

model.compile(loss=Loss,
  optimizer=LossScaleOptimizer(RMSprop(lr=learningRate, epsilon=1e-4), 1000))
# model.compile(loss=Loss, 
#   optimizer=RMSprop(lr=learningRate, epsilon=1e-4))


model.summary()

if not modelFromFile :
  # serialize model to JSON
  model_json = model.to_json()
  with open(resourcesFolder + modelName + ".json", "w") as json_file:
      json_file.write(model_json)
  
  print("Saved model to disk")

if trainModel :
  if trainFromCheckpoint :
    model.load_weights(weightsImport)

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
  model.load_weights(weightsImport)
  if lossGraph :
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

# sampleGenerator = SampleSequence(batchSize, np.arange(startFrame, endFrame), frameShape, GetSampleMaps(frameShape, setDescription, shuffleSeed))

# dataExample = sampleGenerator.__getitem__(0)[0]['input_0'][0]
# frameShape = dataExample.shape

# batchPerFrame = (frameShape[0] * frameShape[1])//batchSize
# if randomSample :
#   testFrame = random.randint(startFrame, endFrame)
#   testBatch = random.randint(0, batchPerFrame)
#   testElement = random.randint(0, batchSize)
# else :
#   testFrame = sample - startFrame
#   testBatch = random.randint(0, batchPerFrame)
#   testElement = random.randint(0, batchSize)

# example = sampleGenerator.__getitem__(testFrame * batchPerFrame + testBatch)

# testPredict = model.predict(example[0])
# testLoss = model.evaluate_generator(testGenerator)

# Display sample results for debugging purpose
# print("Test color : ", testPredict)
# print("Expected color : ", example[1])
# print("Test loss : ", testLoss)

# start = perf_counter_ns()
# batchPredict = model.predict_generator(testGenerator)[testElement]
# end = perf_counter_ns()

# print("Time per image: {:.2f}ms ".format((end-start)/(examplesCount*testSetFraction)/1000000.0))

#-------------------------Test Render--------------------------#

if testRender:
  fig = plt.figure(figsize=(8,8))
  
  render_0FinalImage = imageio.imread('D:/Bachelor_resources/Capture1/Capture1_FinalImage_0839.png')[:,:,:3]
  frameShape = render_0FinalImage.shape

  padSize = math.floor((dataShape - 1)/2)
  render_0SceneColor = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 3))
  render_0SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))
  render_1SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))
  render_2SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))

  render_0SceneColor[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneColor_0839.png')[:,:,:3]/255.0).astype('uint8')
  render_0SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0839.hdr')[:,:,:1]/3000.0).astype('float32')
  render_1SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0838.hdr')[:,:,:1]/3000.0).astype('float32')
  render_2SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0837.hdr')[:,:,:1]/3000.0).astype('float32')
  
  rowSteps = 10
  renderGenerator = MakeRenderGenerator(render_0SceneColor, render_0SceneDepth, render_1SceneDepth, render_2SceneDepth, frameShape, rowSteps)
  renderedImage = model.predict_generator(renderGenerator, steps=frameShape[0] * rowSteps)


  finalImage = np.reshape(renderedImage, frameShape)

  fig.add_subplot(2, 1, 1)
  plt.imshow(render_0FinalImage)

  fig.add_subplot(2, 1, 2)
  plt.imshow(finalImage.astype('uint8'))

  plt.show()

  # Compute pixel loss
  renderLoss = RenderLoss(render_0FinalImage, finalImage)
  baseVariation = RenderLoss(render_0FinalImage, \
    imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneColor_0839.png')[:,:,:3])
  maxLoss = np.amax(renderLoss)

  plt.imshow(renderLoss/maxLoss)
  
  fileNumber = 0
  while (modelName + "_Render_{}.png".format(GetFrameString(fileNumber, 2))) in os.listdir(resourcesFolder + "Renders/"):
    fileNumber += 1
  
  fileNumberString = GetFrameString(fileNumber, 2)

  # Export frame data
  imageio.imwrite(resourcesFolder + "Renders/" + modelName + "_Render_{}.png".format(fileNumberString), finalImage/255)
  imageio.imwrite(resourcesFolder + "Renders/" + modelName + "_LossRender_{}.png".format(fileNumberString), renderLoss/maxLoss)
  imageio.imwrite(resourcesFolder + "Renders/" + modelName + "_BaseVariation_{}.png".format(fileNumberString), renderLoss/maxLoss)
  print("Max loss : {}".format(maxLoss))

#--------------------------------------------------------------#