from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
from functions import Training,\
                      BuildModel,\
                      MakeGenerators,\
                      UpdateGraphData,\
                      RenderImage,\
                      ShowTrainingGraph
from callbacks import MakeCallbacks
import os, pickle
import numpy as np

class SettingsGUI(qtw.QDialog):

  def __init__(self, settings):
    super(SettingsGUI, self).__init__()

    uic.loadUi('UI/SettingsWindow.ui', self)
    self.userSettings = settings

    #------------------PyQt object references---------------#
    self.dialogButton = self.findChild(qtw.QDialogButtonBox, 'buttonBox')

    self.batchSizeSpinBox = self.findChild(qtw.QSpinBox, 'batchSizeSpinBox')
    self.batchSizeSpinBox.setValue(settings['BatchSize'])
    self.learningRateSpinBox = self.findChild(qtw.QDoubleSpinBox, 'learningRateSpinBox')
    self.learningRateSpinBox.setValue(settings['LearningRate'])
    self.strideSpinBox = self.findChild(qtw.QSpinBox, 'strideSpinBox')
    self.strideSpinBox.setValue(settings['Stride'])
    self.seedSpinBox = self.findChild(qtw.QSpinBox, 'seedSpinBox')
    self.seedSpinBox.setValue(settings['ShuffleSeed'])
    self.inWeightsCheckbox = self.findChild(qtw.QCheckBox, 'inWeightsCheckbox')
    self.inWeightsCheckbox.setChecked(settings['inWeights'])
    self.inWeightsLabel = self.findChild(qtw.QLabel, 'inWeightsLabel')
    self.inWeightsInput = self.findChild(qtw.QLineEdit, 'inWeightsInput')
    self.inWeightsInput.setText(settings['inWeightsSuffix'])

    self.inWeightsCheckbox.toggled.connect(self.inWeightsInput.setEnabled)
    self.inWeightsCheckbox.toggled.connect(self.inWeightsInput.setEnabled)

    self.resourcesFolderInput = self.findChild(qtw.QLineEdit, 'resourcesFolderInput')
    self.resourcesFolderInput.setText(settings['ResourcesFolder'])
    self.framesFolderInput = self.findChild(qtw.QLineEdit, 'framesFolderInput')
    self.framesFolderInput.setText(settings['FramesFolder'])
    self.filePrefixInput = self.findChild(qtw.QLineEdit, 'filePrefixInput')
    self.filePrefixInput.setText(settings['FilePrefix'])
    self.firstFrameSpinBox = self.findChild(qtw.QSpinBox, 'firstFrameSpinBox')
    self.firstFrameSpinBox.setValue(settings['FirstFrame'])
    self.lastFrameSpinBox = self.findChild(qtw.QSpinBox, 'lastFrameSpinBox')
    self.lastFrameSpinBox.setValue(settings['LastFrame'])
    self.randomFramesSpinBox = self.findChild(qtw.QSpinBox, 'randomFramesSpinBox')
    self.randomFramesSpinBox.setValue(settings['RandomFrames'])
    self.digitFormatSpinBox = self.findChild(qtw.QSpinBox, 'digitFormatSpinBox')
    self.digitFormatSpinBox.setValue(settings['DigitFormat'])
    self.includeFramesInput = self.findChild(qtw.QLineEdit, 'includeFramesInput')
    self.includeFramesInput.setText(np.array2string(settings['IncludeFrames'],  separator=', ')\
                                    .replace('[', '').replace(']', ''))

    self.rowStepsSpinBox = self.findChild(qtw.QSpinBox, 'rowStepsSpinBox')
    self.rowStepsSpinBox.setValue(settings['RowSteps'])

    self.dialogButton.accepted.connect(self.DialogButtonPressedOK)
    self.dialogButton.rejected.connect(self.DialogButtonPressedCancel)

    self.show()

  def SaveSettings(self):
    self.userSettings['BatchSize'] = self.batchSizeSpinBox.value()
    self.userSettings['LearningRate'] = self.learningRateSpinBox.value()
    self.userSettings['Stride'] = self.strideSpinBox.value()
    self.userSettings['ShuffleSeed'] = self.seedSpinBox.value()

    resourcesInPath = self.resourcesFolderInput.text()
    if os.path.exists(resourcesInPath):
      self.userSettings['ResourcesFolder'] = resourcesInPath
    else:
      print("Folder {} is invalid, reverted to default value".format(resourcesInPath))
    
    framesInPath = self.framesFolderInput.text()
    if os.path.exists(self.userSettings['ResourcesFolder'] + framesInPath):
      self.userSettings['FramesFolder'] = self.framesFolderInput.text()
    else:
      print("No folder {} in the specified resources folder, reverted to default value".format(framesInPath))
    
    self.userSettings['FilePrefix'] = self.filePrefixInput.text()
    self.userSettings['FirstFrame'] = self.firstFrameSpinBox.value()
    self.userSettings['LastFrame'] = self.lastFrameSpinBox.value()
    self.userSettings['RandomFrames'] = self.randomFramesSpinBox.value()
    self.userSettings['DigitFormat'] = self.digitFormatSpinBox.value()
    
    includeFramesIn = self.includeFramesInput.text()
    includeFramesArr = np.asarray(includeFramesIn.replace(' ', '').split(','))
    try:    
      includeFramesArr.astype('uint8')
    except ValueError:
      print("Invalid include frames input, reverted to default value")
    else:
      self.userSettings['IncludeFrames'] = includeFramesArr.astype('uint8')

    self.userSettings['RowSteps'] = self.rowStepsSpinBox.value()

    saveFile = 'UI/UserSettings.dat'
    with open(saveFile, 'wb+') as settingsFile:
      pickle.dump(self.userSettings, settingsFile)
    print("Saved settings to {}".format(saveFile))

  def DialogButtonPressedOK(self):
    self.SaveSettings()
    self.close()

  def DialogButtonPressedCancel(self):
    self.close()











class AppGUI(qtw.QDialog):

  def __init__(self, defaultSettings):
    super(AppGUI, self).__init__()
    uic.loadUi('UI/MainWindow.ui', self)

    #---------------------Load user settings---------------------#
    self.userSettings = defaultSettings

    self.saveFile = 'UI/UserSettings.dat'
    if os.path.exists(self.saveFile):
        with open(self.saveFile, 'rb') as settingsFile:
            self.userSettings = pickle.load(settingsFile)
    #------------------------------------------------------------#

    self.graphFile = self.userSettings['ResourcesFolder'] + 'Graphs/' + self.userSettings['ModelName'] + '_GraphData.dat'
    self.weightsFile = self.userSettings['ResourcesFolder'] + 'Weights/' + self.userSettings['ModelName'] + '_Weights.h5'
      
    self.modelNameInput = self.findChild(qtw.QLineEdit, 'modelNameInput')
    self.modelNameInput.setText(self.userSettings['ModelName'])
    self.KSizeSpinBox = self.findChild(qtw.QSpinBox, 'KSizeSpinBox')
    self.KSizeSpinBox.setValue(self.userSettings['KSize'])
    self.checkpointCheckbox = self.findChild(qtw.QCheckBox, 'trainFromCheckpointCheckbox')
    self.checkpointCheckbox.setChecked(self.userSettings['TrainFromCheckpoint'])
    self.outputFiedl = self.findChild(qtw.QTextBrowser, 'outputField')
    self.settingsButton = self.findChild(qtw.QPushButton, 'settingsButton')
    self.graphButton = self.findChild(qtw.QPushButton, 'showGraphButton')
    self.trainButton = self.findChild(qtw.QPushButton, 'trainButton')
    self.renderButton = self.findChild(qtw.QPushButton, 'renderButton')
    self.show()

    self.graphButton.clicked.connect(self.ShowGraphButtonPressed)
    self.trainButton.clicked.connect(self.TrainButtonPressed)
    self.renderButton.clicked.connect(self.RenderButtonPressed)
    self.settingsButton.clicked.connect(self.SettingsButtonPressed)

  def closeEvent(self, event):
    self.SaveSettings()
    event.accept()

  def SaveSettings(self):
    self.userSettings['ModelName'] = self.modelNameInput.text()
    self.userSettings['KSize'] = self.KSizeSpinBox.value()
    self.userSettings['TrainFromCheckpoint'] = self.checkpointCheckbox.isChecked()

    saveFile = 'UI/UserSettings.dat'
    with open(saveFile, 'wb+') as settingsFile:
      pickle.dump(self.userSettings, settingsFile)
    print("Saved settings to {}".format(saveFile))

  def TrainButtonPressed(self):
    
    self.SaveSettings()

    model = BuildModel(self.userSettings['KSize'], 
                        self.userSettings['ModelName'], 
                        self.userSettings['LearningRate'])

    generators = MakeGenerators(self.userSettings['FirstFrame'],
                                self.userSettings['LastFrame'],
                                self.userSettings['RandomFrames'],
                                self.userSettings['IncludeFrames'],
                                self.userSettings['Stride'],
                                self.userSettings['KSize'],
                                self.userSettings['ResourcesFolder'] + self.userSettings['FramesFolder'],
                                self.userSettings['BatchSize'],
                                self.userSettings['FilePrefix'],
                                self.userSettings['ShuffleSeed'],
                                self.userSettings['DigitFormat'])

    callbacks = MakeCallbacks(self.graphFile,
                              self.weightsFile,
                              self.userSettings['ResourcesFolder'],
                              self.userSettings['ModelName'],
                              UpdateGraphData,
                              generators['TrainGenerator'])

    Training(model, 
            self.userSettings['TrainEpochs'], 
            callbacks, 
            generators['TrainGenerator'], 
            generators['CrossValidGenerator'], 
            self.userSettings['TrainFromCheckpoint'],
            self.weightsFile, 
            self.graphFile)

  def RenderButtonPressed(self):
    self.SaveSettings()

    model = BuildModel(self.userSettings['KSize'], 
                self.userSettings['ModelName'], 
                self.userSettings['LearningRate'])

    RenderImage(model, 
                self.userSettings['ResourcesFolder'], 
                self.userSettings['KSize'], 
                self.userSettings['RowSteps'])

  def ShowGraphButtonPressed(self):
    ShowTrainingGraph(self.graphFile)

  def SettingsButtonPressed(self):
    self.userSettings['ModelName'] = self.modelNameInput.text().replace(' ', '')
    self.userSettings['KSize'] = self.KSizeSpinBox.value()
    self.userSettings['TrainFromCheckpoint'] = self.checkpointCheckbox.isChecked()

    self.settingsGUI = SettingsGUI(self.userSettings)