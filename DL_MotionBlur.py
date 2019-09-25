if __name__=='__main__':
  print("Importing modules...")
  from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
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
  from sampleSequence import SampleSequence, GetFrameString
  from functions import GetSampleMaps,\
                        GetFrameString,\
                        Loss,\
                        RenderImage,\
                        DebugSample,\
                        ShowTrainingGraph,\
                        Training,\
                        UpdateGraphData

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

  # Training
  dataShape = 201
  modelName = "3Depth_K201"
  trainModel = True
  trainFromCheckpoint = True
  batchSize = 128
  trainEpochs = 5
  stride = 10
  learningRate = 0.001
  saveFiles = True

  shuffleSeed = 42

  # Debug & Visualization
  lossGraph = False
  testRender = True
  debugSample = False
  randomSample = False
  sample = 420

  # File handling
  digitFormat = 4
  randomFrames = 20
  startFrame = 228
  endFrame = 963
  resourcesFolder = "C:/Bachelor_resources/"
  workDirectory = resourcesFolder  + 'Capture1_Sorted/'
  filePrefix = 'Capture1_'

  # Model output
  weightsInFile = resourcesFolder + "Weights/" + modelName + "_2_Weights.h5"
  weightsFileName = resourcesFolder + "Weights/" + modelName + "_2_Weights.h5"
  graphDataFileName = resourcesFolder + "Graphs/" + modelName + "_GraphData.dat"

  #------------------------TF session-------------------------#

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.compat.v1.Session(config=config)

  #-----------------------File handling-----------------------#

  np.random.seed(shuffleSeed)
  setDescription = np.random.randint(startFrame, endFrame + 1, randomFrames) # Contains a random sample of frames to use as a data set
  setDescription = np.append(setDescription, [290, 332, 406, 540, 608, 721, 728, 736, 742, 748, 763, 788, 803, 817, 827, 963, 776, 532])
  np.random.shuffle(setDescription)
  setCount = len(setDescription)
  frameShape = imageio.imread(workDirectory + 'SceneDepth/' + filePrefix + 'SceneDepth_' + GetFrameString(setDescription[0], digitFormat) + '.hdr').shape # Test image for shape

  examplesCount = setCount * frameShape[0] * frameShape[1] /stride
  examplesDisplayCount = examplesCount/1000000
  print("\nTotal training examples : {:.2f} Million".format(examplesDisplayCount))
  trainSetFraction = 1
  crossValidSetFraction = 0.2
  testSetFraction = 0.2

  #-----------------------PyQt Setup--------------------------#

  app = QApplication([])
  window = QWidget()
  layout = QVBoxLayout()
  layout.addWidget(QPushButton('Top'))
  layout.addWidget(QPushButton('Bottom'))
  window.setLayout(layout)
  window.show()
  app.exec_()

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

  model.compile(loss=Loss,
    optimizer=LossScaleOptimizer(RMSprop(lr=learningRate, epsilon=1e-4), 1000))

  model.summary()

  #--------------------Data Generators------------------------#

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


  print("\nTraining set size : {:.2f} Million".format(trainGenerator.__len__() * batchSize/1000000))
  print("Cross validation set size : {:.2f} Million".format(crossValidGenerator.__len__() * batchSize/1000000))
  print("Test set size : {:.2f} Million".format(testGenerator.__len__() * batchSize/1000000))
  print()

  #-----------------------Keras Callbacks---------------------#

  class SaveGraphCallback(tf.keras.callbacks.Callback) :
    def __init__(self, graphDataFile, trainGenerator) :
      self.graphDataFile = graphDataFile
      self.trainSetSize = trainGenerator.__len__()

    def on_epoch_end(self, epoch, logs=None) :
      UpdateGraphData(logs['loss'], logs['val_loss'], self.trainSetSize, self.graphDataFile)

  graphDataUpdate = SaveGraphCallback(graphDataFileName, trainGenerator)
  trainCheckpoint = ModelCheckpoint(weightsFileName, verbose=0, save_weights_only=True)
  backupCheckpoint = ModelCheckpoint(resourcesFolder + "Weights/" + modelName + "_{epoch:02d}" + "_Weights.h5", verbose=0, save_weights_only=True)

  #-----------------------------------------------------------#

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
    
  if trainModel :
    Training(model, 
            trainEpochs,
            [trainCheckpoint, backupCheckpoint, SaveGraphCallback],
            trainGenerator,
            crossValidGenerator,
            trainFromCheckpoint,
            weightsInFile,
            weightsFileName,
            graphDataFileName,
            lossGraph)

  elif testRender :
    rowSteps = 20
    RenderImage(model, resourcesFolder, dataShape, rowSteps)
