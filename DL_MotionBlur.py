if __name__=='__main__':
  print("Importing modules...")
  from matplotlib import pyplot as plt
  import numpy as np
  import tensorflow as tf
  from tensorflow.keras.models import load_model
  import os, math, random, imageio, pickle, importlib, sys
  import tensorflow.python.util.deprecation as deprecation
  deprecation._PRINT_DEPRECATION_WARNINGS = False
  from sampleSequence import SampleSequence, GetFrameString
  from PyQt5 import QtWidgets as qtw
  from UI.gui import AppGUI
  from functions import GetSampleMaps,\
                        GetFrameString,\
                        Loss,\
                        RenderImage,\
                        DebugSample,\
                        ShowTrainingGraph,\
                        Training,\
                        UpdateGraphData,\
                        MakeGenerators,\
                        

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
  os.chdir(os.path.dirname(__file__))

  #----------------------Default values-----------------------#
  defaultSettings = {
    'KSize': 201,
    'ModelName': '3Depth_K201',
    'BatchSize': 128,
    'TrainFromCheckpoint': False,
    'TrainEpochs': 5,
    'LearningRate': 0.001,
    'Stride': 10,
    'ShuffleSeed': 36,
    'RandomFrames': 20,
    'ResourcesFolder': 'C:/Bachelor_resources/',
    'FramesFolder': 'Capture1_Sorted',
    'FilePrefix': 'Capture1_',
    'FirstFrame': 228,
    'LastFrame': 963,
    'RandomFrames': 20,
    'DigitFormat': 4,
    'IncludeFrames': [290, 332, 406, 540, 608, 721, 
                      728, 736, 742, 748, 763, 788, 
                      803, 817, 827, 963, 776, 532]}

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

  #-----------------------PyQt Setup--------------------------#

  app = qtw.QApplication(sys.argv)
  window = AppGUI()
  # window.show()
  app.exec_()

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
    
  if trainModel:
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

  elif testRender:
    rowSteps = 20
    RenderImage(model, resourcesFolder, dataShape, rowSteps)
