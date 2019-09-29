if __name__=='__main__':
  print("Importing modules...")
  import numpy as np
  import os, math, random, imageio, pickle, importlib, sys
  import tensorflow.python.util.deprecation as deprecation
  deprecation._PRINT_DEPRECATION_WARNINGS = False
  from sampleSequence import SampleSequence, GetFrameString
  from PyQt5 import QtWidgets as qtw
  from gui import AppGUI
  from functions import GetSampleMaps,\
                        GetFrameString,\
                        DebugSample

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
    'inWeights': True,
    'inWeightsSuffix': '_Weights',
    'ResourcesFolder': 'C:/Bachelor_resources/',
    'FramesFolder': 'Capture1_Sorted',
    'FilePrefix': 'Capture1_',
    'FirstFrame': 228,
    'LastFrame': 963,
    'RandomFrames': 20,
    'DigitFormat': 4,
    'IncludeFrames': np.array([290, 332, 406, 540, 608, 721, 
                              728, 736, 742, 748, 763, 788, 
                              803, 817, 827, 963, 776, 532]),
    'RowSteps': 20}

  DebugSample = False

  #-----------------------PyQt Setup--------------------------#

  app = qtw.QApplication(sys.argv)
  window = AppGUI(defaultSettings)
  window.show()
  app.exec_()
