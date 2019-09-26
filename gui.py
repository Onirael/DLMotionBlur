from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
from functions import Training, BuildModel, MakeGenerators, UpdateGraphData
from callbacks import MakeCallbacks
import os, pickle

class AppGUI(qtw.QDialog):

  def __init__(self, defaultSettings):
    super(AppGUI, self).__init__()
    uic.loadUi('UI/mainWindow.ui', self)

    #---------------------Load user settings---------------------#
    self.userSettings = defaultSettings

    if os.path.exists('UI/UserSettings.dat'):
        with open('UI/UserSettings.dat', 'rb') as settingsFile:
            self.userSettings = pickle.load(settingsFile)

    # self.modelName = self.userSettings['ModelName']
    # self.KSize = self.userSettings['KSize']
    # self.checkpointTraining = self.userSettings['TrainFromCheckpoint']
    # self.batchSize = self.userSettings['BatchSize']
    # self.learningRate = self.userSettings['LearningRate']
    # self.stride = self.userSettings['Stride']
    # self.seed = self.userSettings['ShuffleSeed']
    # self.randomFrames = self.userSettings['RandomFrames']
    # self.resourcesFolder = self.userSettings['ResourcesFolder']
    # self.workDirectory = self.resourcesFolder + self.userSettings['FramesFolder']
    # self.filePrefix = self.userSettings['FilePrefix']
    # self.startFrame = self.userSettings['FirstFrame']
    # self.endFrame = self.userSettings['LastFrame']
    # self.randomFrames = self.userSettings['RandomFrames']
    # self.digitFormat = self.userSettings['DigitFormat']
    # self.includeFrames = self.userSettings['IncludeFrames']
    #------------------------------------------------------------#
      
    self.modelNameInput = self.findChild(qtw.QLineEdit, 'modelNameInput')
    self.KSizeSpinBox = self.findChild(qtw.QSpinBox, 'KSizeSpinBox')
    self.checkpointCheckbox = self.findChild(qtw.QSpinBox, 'trainFromCheckpointCheckbox')
    self.outputFiedl = self.findChild(qtw.QTextBrowser, 'outputField')
    self.settingsButton = self.findChild(qtw.QPushButton, 'settingsButton')
    self.graphButton = self.findChild(qtw.QPushButton, 'showGraphButton')
    self.trainButton = self.findChild(qtw.QPushButton, 'trainButton')
    self.renderButton = self.findChild(qtw.QPushButton, 'renderButton')
    self.show()

    def trainButtonPressed(self):
      model = BuildModel(self.userSettings['KSize'], 
                        self.userSettings['ModelName'], 
                        self.userSettings['LearningRate'])

      generators = MakeGenerators(self.userSettings['FirstFrame'],
                                  self.userSettings['LastFrame'],
                                  self.userSettings['RandomFrames'],
                                  self.userSettings['Stride'],
                                  self.userSettings['KSize'],
                                  self.userSettings['ResourcesFolder'] + self.userSettings['FramesFolder'],
                                  self.userSettings['BatchSize'],
                                  self.userSettings['FilePrefix'],
                                  self.userSettings['ShuffleSeed'],
                                  self.userSettings['DigitFormat'],)

      callbacks = MakeCallbacks(self.graphDataFile,
                                self.weightsFile,
                                self.userSettings['ResourcesFolder'],
                                self.userSettings['ModelName'],
                                UpdateGraphData,
                                generators['TrainGenerator'])

      training =Training(model, 
                        self.userSettings['TrainEpochs'], 
                        callbacks, 
                        generators['TrainGenerator'], 
                        generators['CrossValidGenerator'], 
                        self.userSettings['TrainFromCheckpoint'],
                        weightsFile, 
                        graphDataFile)

    def renderButtonPressed(self):
