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
digitFormat = 5
batchSize = 200
useAllExamples = True
usedExamples = 10000
debugSample = False
sample = 150
shuffleSeed = 42
randomSample = True
trainModel = False
trainEpochs = 50
saveFiles = False
modelFromFile = True
displayData = False
lossGraph = True
resourcesFolder = "D:/Bachelor_resources/"
testRender = True
weightsFileName = resourcesFolder + "2Depth_0_Weights.h5"
modelFileName = resourcesFolder + "2Depth_0_Model.h5"

workDirectory = resourcesFolder  + 'samples3_Capture1/'
filePrefix = 'Capture1_'

#------------------------TF session-------------------------#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#-----------------------Keras Sequence----------------------#

class DataSequence(tf.keras.utils.Sequence) :

  def __init__(self, indexArray, directory, batch_size, verbose=False) :
    self.batch_size = batch_size
    self.directory = directory
    self.indexArray = indexArray
    self.verbose = verbose
    self.arrayLen = len(indexArray)
    self.batchAmount = math.ceil(self.arrayLen/self.batch_size)

  def __len__(self) :
    return self.batchAmount
  
  def __getitem__(self, idx) :
    global dataShape
    global digitFormat
    global filePrefix

    batch_SceneColor = np.zeros((self.batch_size, dataShape, dataShape, 3))
    batch_SceneDepth0 = np.zeros((self.batch_size, dataShape, dataShape, 1))
    batch_SceneDepth1 = np.zeros((self.batch_size, dataShape, dataShape, 1))
    batch_FinalImage = np.zeros((self.batch_size, 3))

    if self.verbose :
      print("\nImporting Batch {}".format(idx))

    for index in range(self.batch_size) :
      if (index + self.batch_size * idx >= self.arrayLen) :
        frame = self.indexArray[index]
      else :
        frame = self.indexArray[index + self.batch_size * idx]

      frameString = str(frame)
      if (len(frameString) < digitFormat) :
        frameString = (digitFormat - len(frameString)) * "0" + frameString

      batch_SceneColor[index] = (imageio.imread(self.directory + 'Input/' + filePrefix + '0SceneColor_' + frameString + '.png')[:,:,:3]/255.0).astype('float16')
      batch_SceneDepth0[index] = (imageio.imread(self.directory + 'Input/' + filePrefix + '0SceneDepth_' + frameString + '.hdr')[:,:,:1]/3000.0).astype('float16')
      batch_SceneDepth1[index] = (imageio.imread(self.directory + 'Input/' + filePrefix + '1SceneDepth_' + frameString + '.hdr')[:,:,:1]/3000.0).astype('float16')
      batch_FinalImage[index] = (imageio.imread(self.directory + 'Output/' + filePrefix + '0FinalImage_' + frameString + '.png')[0,0,:3]).astype('float16')

    return ({'input_0':batch_SceneColor, 'input_1':batch_SceneDepth0, 'input_2':batch_SceneDepth1}, batch_FinalImage)


#-------------------------Functions-------------------------#

def MakeRenderGenerator(sceneColor, sceneDepth0, sceneDepth1, frameShape, rowSteps=4, verbose=True) :

  for row in range(frameShape[0]) :
    if verbose:
      print("Rendering... ({:.2f}%)".format(row/frameShape[0] * 100), end="\r")

    batchSize = math.floor(frameShape[1]/rowSteps)

    for columnStep in range(rowSteps) :
      curRow = \
      {
        'input_0' : np.zeros((batchSize, dataShape, dataShape, 3)),
        'input_1' : np.zeros((batchSize, dataShape, dataShape, 1)),
        'input_2' : np.zeros((batchSize, dataShape, dataShape, 1)),
      }
      for batchColumn in range(batchSize) :
        column = columnStep * batchSize + batchColumn
        curRow['input_0'][batchColumn] = sceneColor[row:dataShape + row, column:dataShape + column]
        curRow['input_1'][batchColumn] = sceneDepth0[row:dataShape + row, column:dataShape + column]
        curRow['input_2'][batchColumn] = sceneDepth1[row:dataShape + row, column:dataShape + column]

      yield curRow

def ApplyKernel(image, flatKernel) :
  global dataShape
  kernel = tf.reshape(flatKernel, [tf.shape(flatKernel)[0], dataShape, dataShape])

  return tf.einsum('hij,hijk->hk', kernel, image)

def Loss(y_true, y_pred) :    # Nested function definition
  delta = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
  return delta

#-----------------------File handling-----------------------#

if useAllExamples :
  setCount = len(os.listdir(workDirectory + "Output"))
  setDescription = np.arange(0, setCount)
else :
  setCount = usedExamples
  # fileCount = len(os.listdir(workDirectory + "Output")) # Only take n first examples while examples are being created
  # setDescription = np.random.choice(fileCount, setCount, replace=False)
  setDescription = np.arange(0, setCount)

print("Total training examples : " + str(setCount) + "\n")

trainingSetSize = math.floor(setCount * 0.6)
crossValidSetSize = math.floor(setCount * 0.2)
testSetSize = math.floor(setCount * 0.2)

np.random.seed(shuffleSeed)
np.random.shuffle(setDescription)
trainSet = setDescription[:trainingSetSize]
crossValidSet = setDescription[trainingSetSize:trainingSetSize + crossValidSetSize]
testSet = setDescription[trainingSetSize + crossValidSetSize:]

trainGenerator = DataSequence(trainSet, workDirectory, batchSize, verbose=False)
crossValidGenerator = DataSequence(crossValidSet, workDirectory, batchSize)
testGenerator = DataSequence(testSet, workDirectory, batchSize)

print("\nTraining set size : ", trainingSetSize)
print("Cross validation set size : ", crossValidSetSize)
print("Test set size : ", testSetSize)
print()

#-------------------------Debug-----------------------------#

if (debugSample) :
  if (randomSample) :
    sample = random.randint(0, len(trainSet))
  plotTitle = "Sample " + str(trainSet[sample])

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
  plt.imshow(example[0]['input_1'][batchElement,:,:,0], cmap='gray')
  fig.add_subplot(2, 2, 3)
  plt.imshow(example[0]['input_2'][batchElement,:,:,0], cmap='gray')
  fig.add_subplot(2, 2, 4)
  plt.imshow(example[1][batchElement]/255.0)


  print("Max depth : ", np.amax(example[0]['input_1'][batchElement]))
  plt.show()
  quit()

#---------------------TensorFlow model----------------------#

input0 = tf.keras.Input(shape=(dataShape, dataShape, 3), name='input_0') #Scene color
input1 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_1') #Depth 0
input2 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_2') #Depth -1

#-Definition---------------------#

#Input1
y = tf.keras.layers.MaxPooling2D(2,2)(input1)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.MaxPooling2D(4,4)(y)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.MaxPooling2D(2,2)(y)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.Flatten()(y)
# y = tf.keras.layers.Dense(dataShape, activation='relu')(y)
y = tf.keras.Model(inputs=input1, outputs=y)

#Input2
z = tf.keras.layers.MaxPooling2D(2,2)(input2)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.MaxPooling2D(4,4)(z)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.MaxPooling2D(2,2)(z)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.Flatten()(z)
# z = tf.keras.layers.Dense(dataShape, activation='relu')(z)
z = tf.keras.Model(inputs=input2, outputs=z)

#Combine inputs
combined = tf.keras.layers.concatenate([y.output, z.output])

#Common network
n = tf.keras.layers.Dense(64, activation='relu')(combined)
n = tf.keras.layers.Dense(dataShape**2, activation='linear')(n)
n = tf.keras.layers.Lambda(lambda x: ApplyKernel(input0, x))(n)

#Model
model = tf.keras.Model(inputs=[input0, y.input, z.input], outputs=n)

#--------------------------------#

model.compile(loss=Loss, 
  optimizer=RMSprop(lr=0.001))

# Save layer names for plotting
layer_names = [layer.name for layer in model.layers]

model.summary()

if (trainModel) :
  training = model.fit_generator(
    trainGenerator,
    validation_data=crossValidGenerator,
    validation_steps=math.ceil(testSetSize/batchSize),
    epochs=trainEpochs,
    steps_per_epoch=math.ceil(trainingSetSize/batchSize),
    # use_multiprocessing=True
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

#--------------------------------------------------------------#

  if (displayData) :
    #-----------------------Display data------------------------#
    sampleSequence = DataSequence(testSet, workDirectory, batchSize)
    example = sampleSequence.__getitem__(math.floor(sample/batchSize))
    x = example[sample%batchSize]

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

else :
  model.load_weights(weightsFileName)


#--------------------------Test Model--------------------------#

testLoss = model.evaluate_generator(testGenerator, steps=math.ceil(testSetSize/batchSize))

sampleSequence = DataSequence(testSet, workDirectory, batchSize)
example = sampleSequence.__getitem__(math.floor(sample/batchSize))
batchElement = sample%batchSize

testPredict = model.predict(example[0])

# Display sample results for debugging purpose
# print("Test color : ", testPredict)
# print("Expected color : ", example[1])
np.set_printoptions(precision=3, suppress=True)
print("Error vectors : \n", np.absolute(example[1] - testPredict)[:10])
print("Test loss : {:.2f}".format(testLoss))

start = perf_counter_ns()
batchPredict = model.predict_generator(testGenerator, steps=math.ceil(testSetSize/batchSize))[batchElement]
end = perf_counter_ns()

print("Time per image: {:.2f}ms ".format((end-start)/testSetSize/1000000.0))

if testRender:
  fig = plt.figure(figsize=(8,8))
  
  render_0FinalImage = imageio.imread('D:/Bachelor_resources/Capture1/Capture1_FinalImage_0411.png')
  frameShape = render_0FinalImage.shape

  padSize = math.floor((dataShape - 1)/2)
  render_0SceneColor = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 3))
  render_0SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))
  render_1SceneDepth = np.zeros((frameShape[0] + 2 * padSize, frameShape[1] + 2 * padSize, 1))

  render_0SceneColor[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneColor_0412.png')[:,:,:3]/255.0).astype('float16')
  render_0SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0412.hdr')[:,:,:1]/3000.0).astype('float16')
  render_1SceneDepth[padSize:padSize + frameShape[0], padSize:padSize + frameShape[1]] = \
    (imageio.imread('D:/Bachelor_resources/Capture1/Capture1_SceneDepth_0411.hdr')[:,:,:1]/3000.0).astype('float16')
  
  rowSteps = 4
  renderGenerator = MakeRenderGenerator(render_0SceneColor, render_0SceneDepth, render_1SceneDepth, frameShape, rowSteps=rowSteps)
  renderedImage = model.predict_generator(renderGenerator, steps=frameShape[0] * rowSteps)

  finalImage = np.reshape(renderedImage/255.0, (frameShape[0], frameShape[1], 3))

  fig.add_subplot(2, 1, 1)
  plt.imshow(render_0FinalImage)

  fig.add_subplot(2, 1, 2)
  plt.imshow(finalImage)

  plt.show()