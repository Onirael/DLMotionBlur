from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from time import perf_counter_ns
import os, math, random, imageio
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

dataShape = 201
frameSizeX = 1920
frameSizeY = 1080
digitFormat = 4
debugSample = False
sample = 110
shuffleSeed = 42
randomSample = False
trainModel = False
trainEpochs = 50
saveFiles = True
modelFromFile = True
displayData = False
lossGraph = True
resourcesFolder = "D:/Bachelor_resources/"
weightsFileName = resourcesFolder + "2Depth_0_Weights.h5"
modelFileName = resourcesFolder + "2Depth_0_Model.h5"
testRender = True
testFrame = resourcesFolder + "Capture1/Capture1_FinalImage_0407.png"


#------------------------TF session-------------------------#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#-------------------------Callback--------------------------#

class accuracyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training !")
      self.model.stop_training = True

callback = accuracyCallback()

#-------------------------Functions-------------------------#

def AddMargin(image, marginSize) :
  marginImage = np.zeros((image.shape[0] + 2 * marginSize, image.shape[1] + 2 * marginSize, image.shape[2]))
  marginImage[marginSize:image.shape[0] + marginSize , marginSize:image.shape[1] + marginSize] = image

  return marginImage 

def SampleImage(image, samplePixel, sampleSize) :
  shape = image.shape
  sample = np.array([])

  if (max(shape) < 2*sampleSize + 1) :
      print("invalid image or sample size ")
      return sample

  sample = image[samplePixel[0] - sampleSize - 1:samplePixel[0] + sampleSize, samplePixel[1] - sampleSize - 1:samplePixel[1] + sampleSize]
  return sample

def ApplyKernel(image, flatKernel) :
  global dataShape
  kernel = tf.reshape(flatKernel, [tf.shape(flatKernel)[0], dataShape, dataShape])

  return tf.einsum('hij,hijk->hk', kernel, image)


def Loss(y_true, y_pred) :    # Nested function definition
  delta = tf.image.total_variation(y_true, tf.reshape(y_pred, tf.reshape(y_pred, (tf.shape(y_pred)[0], frameSizeX, frameSizeY, 3))))
  return delta

#-----------------------File handling-----------------------#

workDirectory = resourcesFolder  + 'Capture1_Sorted/'
filePrefix = 'Capture1'

def FillNameSet(indexArray, SceneColorArr, SceneDepth0Arr, SceneDepth1Arr, FinalImageArr) :
  for index in indexArray :
    frameString = str(index) 
    if (len(frameString) < digitFormat) :
      frameString = (digitFormat - len(frameString)) * "0" + frameString

    preFrameString = str(index-1)
    if (len(preFrameString) < digitFormat) :
      preFrameString = (digitFormat - len(preFrameString)) * "0" + preFrameString

    sceneColor = workDirectory + 'SceneColor/' + filePrefix + '_' + 'SceneColor' + '_' + frameString + '.png'
    sceneDepth0 = workDirectory + 'SceneDepth/' + filePrefix + '_' + 'SceneDepth' + '_' + frameString + '.hdr'
    sceneDepth1 = workDirectory + 'SceneDepth/' + filePrefix + '_' + 'SceneDepth' + '_' + preFrameString + '.hdr'
    finalImage = workDirectory + 'SceneDepth/' + filePrefix + '_' + 'FinalImage' + '_' + preFrameString + '.png'

    SceneColorArr.append(sceneColor)
    SceneDepth0Arr.append(sceneDepth0)
    SceneDepth1Arr.append(sceneDepth1)
    FinalImageArr.append(finalImage)

startFrame = 227
endFrame = 999
setCount = 999 - 227
print("Total training examples : " + str(setCount) + "\n")

#Split into datasets
trainingSetSize = math.floor(setCount * 0.6)
crossValidSetSize = math.floor(setCount * 0.2)
testSetSize = math.floor(setCount * 0.2)

setDescription = np.shuffle(np.arange(startFrame, endFrame + 1))
np.random.seed(shuffleSeed)
np.random.shuffle(setDescription)
trainSet = setDescription[:math.floor(setCount * 0.6)]
crossValidSet = setDescription[math.floor(setCount * 0.6):math.floor(setCount * 0.8)]
testSet = setDescription[math.floor(setCount * 0.8):]

train_SceneColor = []
train_SceneDepth0 = []
train_SceneDepth1 = []
train_FinalImage = []

crossValid_SceneColor = []
crossValid_SceneDepth0 = []
crossValid_SceneDepth1 = []
crossValid_FinalImage = []

test_SceneColor = []
test_SceneDepth0 = []
test_SceneDepth1 = []
test_FinalImage = []

FillNameSet(trainSet, train_SceneColor, train_SceneDepth0, train_SceneDepth1, train_FinalImage)
FillNameSet(crossValidSet, crossValid_SceneColor, crossValid_SceneDepth0, crossValid_SceneDepth1, crossValid_FinalImage)
FillNameSet(testSet, test_SceneColor, test_SceneDepth0, test_SceneDepth1, test_FinalImage)

#----------------------Image processing---------------------#

genColor = ImageDataGenerator(rescale = 1./255)

def GetTrainGenerator(generator, )

print("\nTraining set size : ", len(train_FinalImage))
print("Cross validation set size : ", len(crossValid_FinalImage))
print("Test set size : ", len(test_FinalImage))
print()

#-------------------------Debug-----------------------------#

if (debugSample) :
  if (randomSample) :
    sample = random.randint(0, len(trainingSet_0SceneColor))
  plotTitle = "Sample " + str(sample)

  fig = plt.figure(figsize=(8,8))
  fig.suptitle(plotTitle, fontsize=16)

  fig.add_subplot(2, 2, 1)
  plt.imshow(trainingSet_0SceneColor[sample])
  fig.add_subplot(2, 2, 2)
  plt.imshow(trainingSet_0SceneDepth[sample,:,:,0] * 65280.0, cmap='gray')
  fig.add_subplot(2, 2, 3)
  plt.imshow(trainingSet_1SceneDepth[sample,:,:,0] * 65280.0, cmap='gray')
  fig.add_subplot(2, 2, 4)
  plt.imshow(trainingSet_0FinalImage[sample])

  centerPixel = math.floor((dataShape - 1)/2 + 1)
  print("\nCenter pixel value : ", trainingSet_0SceneColor[sample, centerPixel, centerPixel])
  print("Max depth : ", np.amax(trainingSet_0SceneDepth[sample]))


  plt.show()
  quit()

#---------------------TensorFlow model----------------------#

input0 = tf.keras.Input(shape=(frameSizeX, frameSizeY, 3), name='input_0') #Scene color
input1 = tf.keras.Input(shape=(frameSizeX, frameSizeY, 1), name='input_1') #Depth 0
input2 = tf.keras.Input(shape=(frameSizeX, frameSizeY, 1), name='input_2') #Depth -1

#-Definition---------------------#

#Input0
x = tf.keras.layers.Flatten()(x)
x = tf.keras.Model(inputs=input0, outputs=x)

#Input1
y = tf.keras.layers.MaxPooling2D(2,2)(input1)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.MaxPooling2D(4,4)(y)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.MaxPooling2D(2,2)(y)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.Flatten()(y)
y = tf.keras.layers.Dense(dataShape, activation='relu')(y)
y = tf.keras.Model(inputs=input1, outputs=y)

#Input2
z = tf.keras.layers.MaxPooling2D(2,2)(input2)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.MaxPooling2D(4,4)(z)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.MaxPooling2D(2,2)(z)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.Flatten()(z)
z = tf.keras.layers.Dense(dataShape, activation='relu')(z)
z = tf.keras.Model(inputs=input2, outputs=z)

#Combine inputs
combined = tf.keras.layers.concatenate([y.output, z.output])

#Common network
n = tf.keras.layers.Dense(64, activation='relu')(combined)
n = tf.keras.layers.Dense(dataShape**2, activation='relu')(n)
n = tf.keras.layers.concatenate([n, x.output])
n = tf.keras.layers.Dense(frameSizeX * frameSizeY, activation='relu')(n)

#Model
model = tf.keras.Model(inputs=[input0, y.input, z.input], outputs=n)

#--------------------------------#

model.compile(loss=Loss, 
  optimizer=RMSprop(lr=0.001))

# Save layer names for plotting
layer_names = [layer.name for layer in model.layers]

if (trainModel) :

  model.summary()

  training = model.fit(
    [trainingSet_0SceneColor, trainingSet_0SceneDepth, trainingSet_1SceneDepth],
    trainingSet_0FinalImage,
    validation_data=([crossValidSet_0SceneColor, crossValidSet_0SceneDepth, crossValidSet_1SceneDepth], crossValidSet_0FinalImage),
    epochs=trainEpochs
  )

  # Get training and test loss histories
  training_loss = training.history['loss']
  test_loss = training.history['val_loss']

  epoch_count = range(1, len(training_loss) + 1) # Create count of the number of epochs

  if saveFiles :
    model.save_weights(weightsFileName)
    print("Saved weights to file")

    # model.save(modelFileName)
    # print("Saved model to file")

  if (lossGraph) :
    #-----------------Visualize loss history--------------------#

    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.xlim(0, len(training_loss))
    plt.ylabel('Loss')
    plt.ylim(0, 70)
    plt.show()

else :
  model.load_weights(weightsFileName)

#--------------------------------------------------------------#

  if (displayData) :
    #-----------------------Display data------------------------#

    x = {'input_0':testSet_0SceneColor[np.newaxis, sample], 'input_1':testSet_0SceneDepth[np.newaxis, sample],
      'input_2':testSet_1SceneDepth[np.newaxis, sample]}

    layer_outputs = [layer.output for layer in model.layers[2:8]]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    activation_model.summary()
    activations = activation_model.predict(x)

    fig = plt.figure(figsize=(8,8))

    for layer in activations :
      print(layer.shape)

    fig.add_subplot(6, 16, 1)
    imShape = activations[0][0].shape
    plt.imshow(np.reshape(activations[0][0], (imShape[0], imShape[1])), cmap='gray')
    fig.add_subplot(6, 16, 17)
    imShape = activations[1][0].shape
    plt.imshow(np.reshape(activations[1][0], (imShape[0], imShape[1])), cmap='gray')

    print("Activation length: ", len(activations))
    for j in range(4) :
      for i in range(16) :
        fig.add_subplot(6, 16, 32 + j * 16 + i + 1)
        imShape = activations[j + 2].shape
        plt.imshow(np.reshape(activations[j + 2][:,:,:,i], (imShape[1], imShape[2])), cmap='gray')

    plt.show()

    #-----------------------------------------------------------#

testLoss = model.evaluate({'input_0':testSet_0SceneColor, 'input_1':testSet_0SceneDepth,
 'input_2':testSet_1SceneDepth}, testSet_0FinalImage)

testPredict = model.predict({'input_0':testSet_0SceneColor[np.newaxis, sample], 'input_1':testSet_0SceneDepth[np.newaxis, sample], 
  'input_2':testSet_1SceneDepth[np.newaxis, sample]})

# Display sample results for debugging purpose
print("Test color : ", testPredict)
print("Expected color : ", crossValidSet_0FinalImage[np.newaxis, sample])

start = perf_counter_ns()
batchPredict = model.predict({'input_0':testSet_0SceneColor, 'input_1':testSet_0SceneDepth, 
  'input_2':testSet_1SceneDepth})
end = perf_counter_ns()

print("Time per image: {:.2f}ms ".format((end-start)/len(testSet_0FinalImage)/1000000.0))

if (testRender):
  sampleSize = math.floor((dataShape - 1)/2.0)

  exSceneColor = imageio.imread(testFrame)[:,:,:3]/255.0
  exSceneDepth0 = imageio.imread(testFrame)[:,:,:1]/65280.0
  exSceneDepth1 = imageio.imread(testFrame)[:,:,:1]/65280.0

  frameShape = exSceneColor.shape

  exSceneColor = AddMargin(exSceneColor, sampleSize)
  exSceneDepth0 = AddMargin(exSceneDepth0, sampleSize)
  exSceneDepth1 = AddMargin(exSceneDepth1, sampleSize)
  
  finalImage = np.zeros((frameShape[0], frameShape[1], 3))
  
  print()

  batchSceneColor = np.zeros((frameShape[1], dataShape, dataShape, 3))
  batchSceneDepth0 = np.zeros((frameShape[1], dataShape, dataShape, 1))
  batchSceneDepth1 = np.zeros((frameShape[1], dataShape, dataShape, 1))

  for x in range(0, frameShape[0]):
    print("Render progress : {:.2f} %".format(100 * x/float(frameShape[0])), end='\r')
    for y in range(0, frameShape[1]):
      batchSceneColor[y] = SampleImage(exSceneColor, (x + sampleSize + 1,y + sampleSize + 1), sampleSize)
      batchSceneDepth0[y] = SampleImage(exSceneDepth0, (x + sampleSize + 1,y + sampleSize + 1), sampleSize)
      batchSceneDepth1[y] = SampleImage(exSceneDepth1, (x + sampleSize + 1,y + sampleSize + 1), sampleSize)

    outColor = model.predict({'input_0':batchSceneColor, 'input_1':batchSceneDepth0, 
      'input_2':batchSceneDepth1})
      
    finalImage[x] = outColor

  plt.imshow(finalImage)
  plt.show()