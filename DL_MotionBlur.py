from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import os, math, random, imageio
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataShape = 201
digitFormat = 5
debugSample = False
sample = 110
randomSample = False
trainModel = True
displayData = False
lossGraph = True
modelFileName = "D:/Bachelor_resources/Model_2Depth_0.h5"

#-------------------------Callback--------------------------#

class accuracyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training !")
      self.model.stop_training = True

callback = accuracyCallback()

#-------------------------Functions-------------------------#

def ApplyKernel(image, kernel) :
  return tf.einsum('hij,hijk->hk', kernel, image)

def Pred_loss(image) :
  global dataShape
  with tf.compat.v1.Session() :

    def Loss(y_true, y_pred) :
      k_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], dataShape, dataShape])
      k_pred = ApplyKernel(image, k_pred)

      delta = tf.reduce_mean(tf.abs(y_true - k_pred), axis=1)
      return delta

    return Loss

#-----------------------File handling-----------------------#

workDirectory = 'D:/Bachelor_resources/samples2_Capture1'
inputDirectory = workDirectory + '/' + 'Input'
outputDirectory = workDirectory + '/' + 'Output'
filePrefix = 'Capture1'
inNameBase = inputDirectory + '/' + filePrefix + '_'
outNameBase = outputDirectory + '/' + filePrefix + '_'

setCount = len(os.listdir(outputDirectory))
print("Total training examples : " + str(setCount) + "\n")

#Split into datasets
trainingSetSize = math.floor(setCount * 0.6)
crossValidSetSize = math.floor(setCount * 0.2)
testSetSize = math.floor(setCount * 0.2)

setDescription = np.zeros(trainingSetSize)
setDescription = np.append(setDescription, np.repeat(1, crossValidSetSize))
setDescription = np.append(setDescription, np.repeat(2, testSetSize))
np.random.shuffle(setDescription)

#----------------------Image processing---------------------#

#Training set
trainingSet_0SceneColor = np.zeros((trainingSetSize, dataShape, dataShape, 3))
trainingSet_0SceneDepth = np.zeros((trainingSetSize, dataShape, dataShape, 1))
trainingSet_1SceneDepth = np.zeros((trainingSetSize, dataShape, dataShape, 1))
trainingSet_0FinalImage = np.zeros((trainingSetSize, 3))

#Cross validation set
crossValidSet_0SceneColor = np.zeros((crossValidSetSize, dataShape, dataShape, 3))
crossValidSet_0SceneDepth = np.zeros((crossValidSetSize, dataShape, dataShape, 1))
crossValidSet_1SceneDepth = np.zeros((crossValidSetSize, dataShape, dataShape, 1))
crossValidSet_0FinalImage = np.zeros((crossValidSetSize, 3))

#Test set
testSet_0SceneColor = np.zeros((testSetSize, dataShape, dataShape, 3))
testSet_0SceneDepth = np.zeros((testSetSize, dataShape, dataShape, 1))
testSet_1SceneDepth = np.zeros((testSetSize, dataShape, dataShape, 1))
testSet_0FinalImage = np.zeros((testSetSize, 3))

#Add input sets

(trainCount, crossValidCount, testCount) = (0,0,0)

for fileNum in range(setCount) :
  if (fileNum%100 == 0) :
    stateString = "Importing image " + str(fileNum)
    print(stateString)
  
  ident = setDescription[fileNum]

  #Import X and Y images
  frameString = str(fileNum) 
  if (len(frameString) < digitFormat) :
    frameString = (digitFormat - len(frameString)) * "0" + frameString

  sceneColor0 = imageio.imread(inNameBase + '0SceneColor_' + frameString + '.png')[:,:,:3]/255.0
  sceneDepth0 = imageio.imread(inNameBase + '0SceneDepth_' + frameString + '.hdr')[:,:,:1]/65280.0
  sceneDepth1 = imageio.imread(inNameBase + '1SceneDepth_' + frameString + '.hdr')[:,:,:1]/65280.0
  finalImage0 = imageio.imread(outNameBase + '0FinalImage_' + frameString + '.png')[0,0,:3]

  if ident == 0 :

    trainingSet_0SceneColor[trainCount] = sceneColor0
    trainingSet_0SceneDepth[trainCount] = sceneDepth0
    trainingSet_1SceneDepth[trainCount] = sceneDepth1
    trainingSet_0FinalImage[trainCount] = finalImage0

    trainCount += 1

  elif ident == 1 :

    crossValidSet_0SceneColor[crossValidCount] = sceneColor0
    crossValidSet_0SceneDepth[crossValidCount] = sceneDepth0
    crossValidSet_1SceneDepth[crossValidCount] = sceneDepth1
    crossValidSet_0FinalImage[crossValidCount] = finalImage0

    crossValidCount += 1

  elif ident == 2 :

    testSet_0SceneColor[testCount] = sceneColor0
    testSet_0SceneDepth[testCount] = sceneDepth0
    testSet_1SceneDepth[testCount] = sceneDepth1
    testSet_0FinalImage[testCount] = finalImage0

    testCount += 1

print("\nTraining set size : ", len(trainingSet_0FinalImage))
print("Cross validation set size : ", len(crossValidSet_0FinalImage))
print("Test set size : ", len(testSet_0FinalImage))
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
  plt.imshow(trainingSet_0FinalImage[sample, np.newaxis, np.newaxis, :])

  centerPixel = math.floor((dataShape - 1)/2 + 1)
  print("\nCenter pixel value : ", trainingSet_0SceneColor[sample, centerPixel, centerPixel])
  print("Max depth : ", np.amax(trainingSet_0SceneDepth[sample]))


  plt.show()
  quit()

#---------------------TensorFlow model----------------------#

input0 = tf.keras.Input(shape=(dataShape, dataShape, 3), name='input_0') #Scene color
input1 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_1') #Depth 0
input2 = tf.keras.Input(shape=(dataShape, dataShape, 1), name='input_2') #Depth -1

#-Definition---------------------#

# #Input0
# x = tf.keras.layers.Dense(16, activation='relu')(input0)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.Model(inputs=input0, outputs=x)

#Input1
y = tf.keras.layers.MaxPooling2D(2,2)(input1)
y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
y = tf.keras.layers.Dense(16, activation='relu')(y)
y = tf.keras.layers.Flatten()(y)
y = tf.keras.layers.Dense(dataShape, activation='relu')(y)
y = tf.keras.Model(inputs=input1, outputs=y)

#Input2
z = tf.keras.layers.MaxPooling2D(2,2)(input2)
z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
z = tf.keras.layers.Dense(16, activation='relu')(z)
z = tf.keras.layers.Flatten()(z)
z = tf.keras.layers.Dense(dataShape, activation='relu')(z)
z = tf.keras.Model(inputs=input2, outputs=z)

#Combine inputs
combined = tf.keras.layers.concatenate([y.output, z.output])

#Common network
n = tf.keras.layers.Dense(dataShape, activation='relu')(combined)
n = tf.keras.layers.Dense(dataShape**2, activation='linear')(n)

#Model
model = tf.keras.Model(inputs=[input0, y.input, z.input], outputs=n)

#--------------------------------#

model.compile(loss=Pred_loss(input0), 
  optimizer=RMSprop(lr=0.001))

# Save layer names for plotting
layer_names = [layer.name for layer in model.layers]

if (trainModel) :

  model.summary()

  training = model.fit(
    [trainingSet_0SceneColor, trainingSet_0SceneDepth, trainingSet_1SceneDepth],
    trainingSet_0FinalImage,
    validation_data=([crossValidSet_0SceneColor, crossValidSet_0SceneDepth, crossValidSet_1SceneDepth], crossValidSet_0FinalImage),
    epochs=20
  )

  # Get training and test loss histories
  training_loss = training.history['loss']
  test_loss = training.history['val_loss']

  epoch_count = range(1, len(training_loss) + 1) # Create count of the number of epochs

  # # Save model and architecture to single file
  # model.save(modelFileName)
  # print("Saved model to disk")

  model.save_weights(modelFileName)
  print("Saved weights to file")

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
  # # Load model
  # input0 = tf.keras.Input(shape=(dataShape, dataShape, 3)) #Scene color
  # model = load_model(modelFileName, custom_objects={'Loss':Pred_loss(input0)})
  # print("Model inputs: ", model.inputs)
  # print("Model outputs: ", model.outputs)

  model.load_weights(modelFileName)
  print("\nWeights loaded from file\n")
  model.summary()

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

testLoss = model.evaluate({'input_0':testSet_0SceneColor[np.newaxis, sample], 'input_1':testSet_0SceneDepth[np.newaxis, sample],
 'input_2':testSet_1SceneDepth[np.newaxis, sample]}, testSet_0FinalImage[np.newaxis, sample])

testPredict = model.predict({'input_0':testSet_0SceneColor[np.newaxis, sample], 'input_1':testSet_0SceneDepth[np.newaxis, sample], 
  'input_2':testSet_1SceneDepth[np.newaxis, sample]})

testColor = np.einsum('hij,hijk->hk', np.reshape(testPredict, (1, dataShape, dataShape)), crossValidSet_0SceneColor[np.newaxis, sample])

# Display sample results for debugging purpose
print("Test color : ", testColor)
print("Expected color : ", crossValidSet_0FinalImage[np.newaxis, sample])