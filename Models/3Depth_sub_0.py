import tensorflow as tf
from functions import ApplyKernel

def MakeModel(inputs, dataShape, modelName) :

  #-Definition---------------------#

  #Subtract0
  x = tf.keras.layers.subtract([inputs[2], inputs[1]])
  x = tf.keras.layers.MaxPooling2D(2,2)(x)
  x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
  x = tf.keras.layers.Conv2D(8, (3,3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(4,4)(x)
  x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
  x = tf.keras.layers.Conv2D(8, (3,3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)
  x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
  x = tf.keras.layers.Conv2D(8, (3,3), activation='relu')(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.Model(inputs=[inputs[2], inputs[1]], outputs=x)

  #Subtract1
  y = tf.keras.layers.subtract([inputs[3], inputs[2]])
  y = tf.keras.layers.MaxPooling2D(2,2)(y)
  y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
  y = tf.keras.layers.Conv2D(8, (3,3), activation='relu')(y)
  y = tf.keras.layers.MaxPooling2D(4,4)(y)
  y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
  y = tf.keras.layers.Conv2D(8, (3,3), activation='relu')(y)
  y = tf.keras.layers.MaxPooling2D(2,2)(y)
  y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
  y = tf.keras.layers.Conv2D(8, (3,3), activation='relu')(y)
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(64, activation='relu')(y)
  y = tf.keras.Model(inputs=[inputs[3], inputs[2]], outputs=y)

  #Combine inputs
  combined = tf.keras.layers.concatenate([x.output, y.output])

  #Common network
  n = tf.keras.layers.Dense(256, activation='relu')(combined)
  n = tf.keras.layers.Dense(256, activation='relu')(combined)
  n = tf.keras.layers.Dense(dataShape**2, activation='linear')(n)
  n = tf.keras.layers.ReLU()(n)
  n = tf.keras.layers.Lambda(lambda l: ApplyKernel(inputs[0], l, dataShape))(n)

  #Model
  model = tf.keras.Model(inputs=[inputs[0], inputs[1], inputs[2], inputs[3]], outputs=n, name=modelName)

  return model