import math, imageio, os, random, pickle, importlib
import numpy as np
import tensorflow as tf
from time import perf_counter_ns
from matplotlib import pyplot as plt
from sampleSequence import SampleSequence, RenderSequence, GetFrameString
from tensorflow.keras.mixed_precision.experimental import LossScaleOptimizer
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

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

def ApplyKernel(image, flatKernel, dataShape) : # Applies convolution kernel to same shaped image
  kernel = tf.reshape(flatKernel, [tf.shape(flatKernel)[0], dataShape, dataShape])

  return tf.einsum('hij,hijk->hk', kernel, image)

def Loss(y_true, y_pred) : # Basic RGB color distance
  delta = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
  return delta

def RenderLoss(y_true, y_pred) :
  delta = np.mean(np.absolute(y_true - y_pred), axis=2)
  return delta

def RenderImage(model, resourcesFolder, dataShape, rowSteps) :

  modelName = model.name
  fig = plt.figure(figsize=(8,8))
  
  render_0FinalImage = imageio.imread('D:/Bachelor_resources/Capture1/Capture1_FinalImage_0839.png')[:,:,:3]
  frameShape = render_0FinalImage.shape

  padSize = math.floor((dataShape - 1)/2)
  render_0SceneColor = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 3))
  render_0SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))
  render_1SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))
  render_2SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))

  render_0SceneColor[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneColor_0839.png')[:,:,:3]/255.0).astype('float16')
  render_0SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0839.hdr')[:,:,:1]/3000.0).astype('float16')
  render_1SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0838.hdr')[:,:,:1]/3000.0).astype('float16')
  render_2SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0837.hdr')[:,:,:1]/3000.0).astype('float16')

  renderGenerator = RenderSequence(render_0SceneColor,
                                    render_0SceneDepth, 
                                    render_1SceneDepth, 
                                    render_2SceneDepth, 
                                    frameShape,
                                    dataShape,
                                    rowSteps)
  
  start = perf_counter_ns()
  renderedImage = model.predict_generator(renderGenerator,
                                          workers=8,
                                          use_multiprocessing=False,
                                          max_queue_size=20)
  end = perf_counter_ns()
  print("Time per sample: {:.2f}ms ".format((end-start)/(renderedImage.shape[0] * renderedImage.shape[1] *1000000.0)))


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

  plt.imshow((255 * renderLoss/maxLoss).astype('uint8'))
  plt.show()
  
  fileNumber = 0
  while (modelName + "_Render_{}.png".format(GetFrameString(fileNumber, 2))) in os.listdir(resourcesFolder + "Renders/"):
    fileNumber += 1
  
  fileNumberString = GetFrameString(fileNumber, 2)

  # Export frame data
  imageio.imwrite(resourcesFolder + "Renders/" + modelName + "_Render_{}.png".format(fileNumberString), finalImage.astype('uint8'))
  imageio.imwrite(resourcesFolder + "Renders/" + modelName + "_LossRender_{}.png".format(fileNumberString), (255 * renderLoss/maxLoss).astype('uint8'))
  imageio.imwrite(resourcesFolder + "Renders/" + modelName + "_BaseVariation_{}.png".format(fileNumberString), (255 * baseVariation/maxLoss).astype('uint8'))
  print("Max loss : {}".format(maxLoss))
  
  with open(resourcesFolder + "Renders/renderLoss.txt", 'a+') as lossFile :
    lossFile.write("\n\n{}: {}".format(modelName, maxLoss))


def DebugSample(batchSize, stride, frameShape, setDescription, randomFrame, dataShape, filePrefix, digitFormat, workDirectory, shuffleSeed, sample=839) :

  dataSampleMaps = GetSampleMaps(frameShape, setDescription, shuffleSeed)
  sampleGenerator = SampleSequence(batchSize, 
                                  setDescription, 
                                  frameShape, 
                                  dataSampleMaps, 
                                  dataShape, filePrefix, digitFormat, workDirectory,
                                  stride=1)

  dataBatch = sampleGenerator.__getitem__(0)
  dataExample = dataBatch[0]['input_0'][0]
  frameShape = dataExample.shape

  batchPerFrame = (frameShape[0] * frameShape[1])//(batchSize * stride)
  if randomFrame :
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


def ShowTrainingGraph(graphDataFileName) :

  if os.path.exists(graphDataFileName) :
    with open(graphDataFileName, 'rb') as graphDataFile :
      training_loss, test_loss, epoch_count, trainSetSize = pickle.load(graphDataFile)

    epochs = range(1, epoch_count)
    plt.title("Training examples : {}".format(trainSetSize))
    plt.plot(epochs, training_loss, 'r--')
    plt.plot(epochs, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.xlim(1, len(training_loss))
    plt.ylabel('Loss')
    plt.ylim(0, 20)
    plt.xticks(np.arange(1, len(training_loss) + 1))
    plt.show()

  else :
    print("\nInvalid graph data file name")

def UpdateGraphData(trainLoss, testLoss, trainSetSize, graphDataFile) :
  training_loss = []
  test_loss = []
  epoch_count = 1
  trainSetSize = [trainSetSize]

  if os.path.exists(graphDataFile) :
    with open(graphDataFile, 'rb') as graphDataFile :
      _training_loss, _test_loss, _epoch_count, _trainSetSize = pickle.load(graphDataFile)

    training_loss = _training_loss + training_loss
    test_loss = _test_loss + test_loss
    epoch_count = _epoch_count + 1
    trainSetSize = _trainSetSize + trainSetSize

  else :
    training_loss = [trainLoss]
    test_loss = [testLoss]

  with open(graphDataFile, 'wb') as graphDataFile :
    pickle.dump((training_loss, test_loss, epoch_count, trainSetSize), graphDataFile)


def Training(model, trainEpochs, callbacks, trainGenerator, crossValidGenerator, 
            trainFromCheckpoint, weightsFile, graphDataFile) :

  if trainFromCheckpoint :
    model.load_weights(weightsFile)

  training = model.fit_generator(
    trainGenerator,
    validation_data=crossValidGenerator,
    epochs=trainEpochs,
    callbacks=callbacks,
    workers=8,
    max_queue_size=20,
    use_multiprocessing=False,
  )

  model.save_weights(weightsFile)
  print("Saved weights to file")
    
  return training

def MakeGenerators(startFrame, endFrame, randomFrames, includeFrames, stride, dataShape, workDirectory, batchSize, filePrefix, seed, digitFormat, verbose=True):
  
  np.random.seed(seed)
  setDescription = np.random.randint(startFrame, endFrame + 1, randomFrames) # Contains a random sample of frames to use as a data set
  setDescription = np.append(setDescription, includeFrames)
  np.random.shuffle(setDescription)
  setCount = len(setDescription)
  frameShape = imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(setDescription[0], digitFormat) + '.hdr').shape # Test image for shape

  examplesCount = setCount * frameShape[0] * frameShape[1] /stride

  if verbose:
    examplesDisplayCount = examplesCount/1000000
    print("\nTotal training examples : {:.2f} Million".format(examplesDisplayCount))

  trainSetFraction = 1
  crossValidSetFraction = 0.2
  testSetFraction = 0.2
  
  #--------------------Data Generators------------------------#

  trainGenerator = SampleSequence(batchSize, 
                                  setDescription, 
                                  frameShape,
                                  GetSampleMaps(frameShape, setDescription, seed),
                                  dataShape, filePrefix, digitFormat, workDirectory,
                                  stride=int(stride//trainSetFraction))

  crossValidGenerator = SampleSequence(batchSize, 
                                      setDescription, 
                                      frameShape,
                                      GetSampleMaps(frameShape, setDescription, seed + 10), 
                                      dataShape, filePrefix, digitFormat, workDirectory,
                                      stride=int(stride//crossValidSetFraction))

  testGenerator = SampleSequence(batchSize, 
                                setDescription, 
                                frameShape,
                                GetSampleMaps(frameShape, setDescription, seed + 20),
                                dataShape, filePrefix, digitFormat, workDirectory,
                                stride=int(stride//testSetFraction))

  if verbose:
    print("\nTraining set size : {:.2f} Million".format(trainGenerator.__len__() * batchSize/1000000))
    print("Cross validation set size : {:.2f} Million".format(crossValidGenerator.__len__() * batchSize/1000000))
    print("Test set size : {:.2f} Million".format(testGenerator.__len__() * batchSize/1000000))
    print()

  return {'TrainGenerator': trainGenerator, 
          'CrossValidGenerator': crossValidGenerator, 
          'TestGenerator': testGenerator}


def BuildModel(dataShape, modelName, learningRate):

  K.set_floatx('float16')
  K.set_epsilon(1e-4)

  input0 = tf.keras.Input(shape=(dataShape, dataShape, 3), name='input_0', dtype='float16') #Scene color
  input1 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_1', dtype='float16') #Depth 0
  input2 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_2', dtype='float16') #Depth -1
  input3 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_3', dtype='float16') #Depth -2

  modelFunc = importlib.import_module('Models.' + modelName)

  model = modelFunc.MakeModel([input0, input1, input2, input3], dataShape, modelName)
  print("Loaded model from disk")

  model.compile(loss=Loss,
    optimizer=LossScaleOptimizer(RMSprop(lr=learningRate, epsilon=1e-4), 1000))

  model.summary()

  return model
