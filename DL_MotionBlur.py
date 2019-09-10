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
batchSize = 20
digitFormat = 4
debugSample = False
sample = 2
shuffleSeed = 42
randomSample = False
trainModel = True
trainEpochs = 50
saveFiles = True
modelFromFile = False
displayData = False
lossGraph = True
resourcesFolder = "D:/Bachelor_resources/"
weightsFileName = resourcesFolder + "2Depth_0_Weights.h5"
modelFileName = resourcesFolder + "2Depth_0_Model.h5"
testRender = True
testFrame = resourcesFolder + "Capture1/Capture1_FinalImage_0407.png"
workDirectory = resourcesFolder  + 'Capture1_Sorted/'
filePrefix = 'Capture1'

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

def MakeGenerator(indexArray, directory, batch_size=50) :
  global filePrefix
  global frameSizeX
  global frameSizeY
  
  arrayLen = len(indexArray)
  batchAmount = math.ceil(arrayLen/batch_size)

  for batch in range(batchAmount) :
    batch_SceneColor = np.zeros((batch_size, frameSizeY, frameSizeX, 3))
    batch_SceneDepth0 = np.zeros((batch_size, frameSizeY, frameSizeX, 1))
    batch_SceneDepth1 = np.zeros((batch_size, frameSizeY, frameSizeX, 1))
    batch_FinalImage = np.zeros((batch_size, frameSizeY, frameSizeX, 3))

    print("Importing Batch {} \r".format(batch))

    for index in range(batch_size) :
      if (index + batch_size * batch >= arrayLen) :
        frame = indexArray[index]
      else :
        frame = indexArray[index + batch_size * batch]

      frameString = str(frame)
      if (len(frameString) < digitFormat) :
        frameString = (digitFormat - len(frameString)) * "0" + frameString

      preFrameString = str(frame - 1)
      if (len(preFrameString) < digitFormat) :
        preFrameString = (digitFormat - len(preFrameString)) * "0" + preFrameString

      batch_SceneColor[index] = imageio.imread(directory + 'SceneColor/' + filePrefix + '_' + 'SceneColor' + '_' + frameString + '.png')[:,:,:3]/255.0
      batch_SceneDepth0[index] = imageio.imread(directory + 'SceneDepth/' + filePrefix + '_' + 'SceneDepth' + '_' + frameString + '.hdr')[:,:,:1]/3000.0
      batch_SceneDepth1[index] = imageio.imread(directory + 'SceneDepth/' + filePrefix + '_' + 'SceneDepth' + '_' + preFrameString + '.hdr')[:,:,:1]/3000.0
      batch_FinalImage[index] = imageio.imread(directory + 'FinalImage/' + filePrefix + '_' + 'FinalImage' + '_' + preFrameString + '.png')[:,:,:3]/255.0

    yield ({'input_0':batch_SceneColor, 'input_1':batch_SceneDepth0, 'input_2':batch_SceneDepth1}, batch_FinalImage)

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
  global frameSizeX
  global frameSizeY
  delta = tf.math.reduce_sum(tf.math.abs(y_true - y_pred)) / (frameSizeX * frameSizeY * 3)
  return delta

#-----------------------File handling-----------------------#

startFrame = 227
endFrame = 999
setCount = 999 - 227
print("Total training examples : " + str(setCount) + "\n")

#Split into datasets
trainingSetSize = math.floor(setCount * 0.6)
crossValidSetSize = math.floor(setCount * 0.2)
testSetSize = math.floor(setCount * 0.2)

setDescription = np.arange(startFrame, endFrame + 1)
np.random.seed(shuffleSeed)
np.random.shuffle(setDescription)
trainSet = setDescription[:math.floor(setCount * 0.6)]
crossValidSet = setDescription[math.floor(setCount * 0.6):math.floor(setCount * 0.8)]
testSet = setDescription[math.floor(setCount * 0.8):]

trainGenerator = MakeGenerator(trainSet, workDirectory, batchSize)
crossValidGenerator = MakeGenerator(crossValidSet, workDirectory, batchSize)
testGenerator = MakeGenerator(testSet, workDirectory, batchSize)

#----------------------Image processing---------------------#

print("Training set size : ", len(trainSet))
print("Cross validation set size : ", len(crossValidSet))
print("Test set size : ", len(testSet))
print()

#-------------------------Debug-----------------------------#

if (debugSample) :
  if (randomSample) :
    sample = random.randint(0, len(trainSet))
  plotTitle = "Sample " + str(sample)

  fig = plt.figure(figsize=(8,8))
  fig.suptitle(plotTitle, fontsize=16)

  sampleGenerator = MakeGenerator(trainSet, workDirectory, batchSize)
  for i in range(math.floor(sample/batchSize) - 1) :
    next(sampleGenerator)
  example = next(sampleGenerator)

  batchElement = sample%batchSize

  fig.add_subplot(2, 2, 1)
  plt.imshow(example[0]['input_0'][batchElement])
  fig.add_subplot(2, 2, 2)
  plt.imshow(example[0]['input_1'][batchElement] * 5000.0, cmap='gray')
  fig.add_subplot(2, 2, 3)
  plt.imshow(example[0]['input_2'][batchElement] * 5000.0, cmap='gray')
  fig.add_subplot(2, 2, 4)
  plt.imshow(example[1][batchElement])


  print("Max depth : ", np.amax(example[0]['input1'][batchElement]))
  plt.show()
  quit()

#---------------------TensorFlow model----------------------#

input0 = tf.keras.Input(shape=(frameSizeY, frameSizeX, 3), name='input_0') #Scene color
input1 = tf.keras.Input(shape=(frameSizeY, frameSizeX, 1), name='input_1') #Depth 0
input2 = tf.keras.Input(shape=(frameSizeY, frameSizeX, 1), name='input_2') #Depth -1

#-Definition---------------------#

#Input0
x = tf.keras.layers.Flatten()(input0)
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
n = tf.keras.layers.Dense(frameSizeY * frameSizeX * 3, activation='relu')(n)
n = tf.keras.layers.Lambda(lambda x : tf.reshape(x, (frameSizeY, frameSizeX, 3)))(n)

#Model
model = tf.keras.Model(inputs=[input0, y.input, z.input], outputs=n)

#--------------------------------#

model.compile(loss=Loss, 
  optimizer=RMSprop(lr=0.001))

# Save layer names for plotting
layer_names = [layer.name for layer in model.layers]

if (trainModel) :

  model.summary()

  training = model.fit_generator(trainGenerator,
    validation_data=(crossValidGenerator),
    validation_steps=math.ceil(crossValidSetSize/batchSize),
    epochs=trainEpochs,
    steps_per_epoch=math.ceil(trainingSetSize/batchSize),
    use_multiprocessing=False,
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

    for i in range(sample - 1):
      next(testGenerator)
    x = next(testGenerator)

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

testLoss = model.evaluate_generator(testGenerator)

sampleGenerator = MakeGenerator(testSet, workDirectory, batchSize)
sampleBatch = math.floor(testSetSize/batchSize)
for i in range(sampleBatch - 1):
  next(sampleGenerator)
x = next(sampleGenerator)[sample%batchSize]

start = perf_counter_ns()
testPredict = model.predict(x)
end = perf_counter_ns()

print("Computation time: {:.2f}ms ".format((end-start)/1000000.0))

plt.imshow(testPredict)
plt.show()