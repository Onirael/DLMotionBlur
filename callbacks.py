from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

def MakeCallbacks(graphFile, weightsFile, resourcesFolder, modelName, UpdateGraphData, trainGenerator):
	class SaveGraphCallback(tf.keras.callbacks.Callback):
	  def __init__(self, graphDataFile, trainGenerator):
	    self.graphDataFile = graphDataFile
	    self.trainSetSize = trainGenerator.__len__()

	  def on_epoch_end(self, epoch, logs=None):
	    UpdateGraphData(logs['loss'], logs['val_loss'], self.trainSetSize, self.graphDataFile)

	graphDataUpdate = SaveGraphCallback(graphFile, trainGenerator)
	trainCheckpoint = ModelCheckpoint(weightsFile, verbose=0, save_weights_only=True)
	backupCheckpoint = ModelCheckpoint(resourcesFolder + "Weights/" + modelName + "_{epoch:02d}" + "_Weights.h5", verbose=0, save_weights_only=True)

	return [graphDataUpdate, trainCheckpoint, backupCheckpoint]