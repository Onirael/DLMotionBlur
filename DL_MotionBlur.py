print("Importing modules...")
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision.experimental import LossScaleOptimizer
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import os, math, random, imageio, pickle, importlib
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from functions import GetSampleMaps,\
                      GetFrameString,\
                      Loss,\
                      RenderImage,\
                      DebugSample
from sampleSequence import SampleSequence, GetFrameString

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

dataShape = 201 # Convolution K size

# Training
trainModel = True
trainFromCheckpoint = False
batchSize = 128
trainEpochs = 5
stride = 10
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
modelName = "3Depth_K201"

weightsInFile = resourcesFolder + "Weights/" + modelName + "_Weights.h5"
weightsFileName = resourcesFolder + "Weights/" + modelName + "2_Weights.h5"
graphDataFileName = resourcesFolder + "Graphs/" + modelName + "_GraphData.dat"

#------------------------TF session-------------------------#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#-----------------------Keras Callback----------------------#

trainCheckpoint = ModelCheckpoint(weightsFileName, verbose=0, save_weights_only=True)
backupCheckpoint = ModelCheckpoint(resourcesFolder + "Weights/" + modelName + "_{epoch:02d}" + "_Weights.h5", verbose=0, save_weights_only=True)

#-----------------------File handling-----------------------#

np.random.seed(shuffleSeed)
setDescription = np.random.randint(startFrame, endFrame, setCount) # Contains a random sample of frames to use as a data set
setDescription = np.append(setDescription, [291, 335, 412, 550, 623, 742, 749, 760, 766, 772, 787, 813, 830, 844, 856, 999, 800, 541])
np.random.shuffle(setDescription)
setCount = len(setDescription)
frameShape = imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(setDescription[0], digitFormat) + '.hdr').shape # Test image for shape

examplesCount = setCount * frameShape[0] * frameShape[1] /stride
examplesDisplayCount = examplesCount/1000000
print("\nTotal training examples : {:.2f} Million".format(examplesDisplayCount))
trainSetFraction = 1
crossValidSetFraction = 0.2
testSetFraction = 0.2

#--------------------Create Generators----------------------#
print("\nGenerating trainig data...")


trainGenerator = SampleSequence(batchSize, 
                                setDescription, 
                                frameShape,
                                GetSampleMaps(frameShape, setDescription, shuffleSeed),
                                dataShape, filePrefix, digitFormat, workDirectory,
                                stride=int(stride//trainSetFraction))

crossValidGenerator = SampleSequence(batchSize, 
                                    setDescription, 
                                    frameShape,
                                    GetSampleMaps(frameShape, setDescription, shuffleSeed + 10), 
                                    dataShape, filePrefix, digitFormat, workDirectory,
                                    stride=int(stride//crossValidSetFraction))

testGenerator = SampleSequence(batchSize, 
                              setDescription, 
                              frameShape,
                              GetSampleMaps(frameShape, setDescription, shuffleSeed + 20),
                              dataShape, filePrefix, digitFormat, workDirectory,
                              stride=int(stride//testSetFraction))


print("\nTraining set size : {:.2f} Million".format(trainGenerator.__len__()))
print("Cross validation set size : {:.2f} Million".format(crossValidGenerator.__len__()))
print("Test set size : {:.2f} Million".format(testGenerator.__len__()))
print()


#-------------------------Debug-----------------------------#

if debugSample:
  DebugSample(batchSize, 
              stride, 
              frameShape, 
              setDescription, 
              randomSample, 
              dataShape, 
              filePrefix, 
              digitFormat,
              workDirectory,
              shuffleSeed)
  quit()

#---------------------TensorFlow model----------------------#

K.set_floatx('float16')
K.set_epsilon(1e-4)

input0 = tf.keras.Input(shape=(dataShape, dataShape, 3), name='input_0', dtype='float16') #Scene color
input1 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_1', dtype='float16') #Depth 0
input2 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_2', dtype='float16') #Depth -1
input3 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_3', dtype='float16') #Depth -2

modelFunc = importlib.import_module('Models.' + modelName)
model = modelFunc.MakeModel([input0, input1, input2, input3], dataShape, modelName)
print("Loaded model from disk")


# model.compile(loss=Loss, 
#   optimizer=RMSprop(lr=learningRate, epsilon=1e-4))

model.compile(loss=Loss,
  optimizer=LossScaleOptimizer(RMSprop(lr=learningRate, epsilon=1e-4), 1000))

model.summary()

if trainModel :
  if trainFromCheckpoint :
    model.load_weights(weightsInFile)
  
  training = model.fit_generator(
    trainGenerator,
    validation_data=crossValidGenerator,
    epochs=trainEpochs,
    callbacks=[trainCheckpoint],
    workers=8,
    use_multiprocessing=False,
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
  model.load_weights(weightsInFile)
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

#-------------------------Test Render--------------------------#

if testRender:
  RenderImage(model, resourcesFolder, dataShape)