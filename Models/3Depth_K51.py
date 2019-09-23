import tensorflow as tf

def MakeModel(inputs, dataShape, modelName, ApplyKernel) :
  #-Definition---------------------#

  #Input1
  x = tf.keras.layers.MaxPooling2D(2,2)(inputs[1])
  x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)
  x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
  # x = tf.keras.layers.MaxPooling2D(2,2)(x)
  # x = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.Model(inputs=inputs[1], outputs=x)

  #Input2
  y = tf.keras.layers.MaxPooling2D(2,2)(inputs[2])
  y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
  y = tf.keras.layers.MaxPooling2D(2,2)(y)
  y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
  # y = tf.keras.layers.MaxPooling2D(2,2)(y)
  # y = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(y)
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.Model(inputs=inputs[2], outputs=y)

  #Input3
  z = tf.keras.layers.MaxPooling2D(2,2)(inputs[3])
  z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
  z = tf.keras.layers.MaxPooling2D(2,2)(z)
  z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
  # z = tf.keras.layers.MaxPooling2D(2,2)(z)
  # z = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(z)
  z = tf.keras.layers.Flatten()(z)
  z = tf.keras.Model(inputs=inputs[3], outputs=z)

  #Combine inputs
  combined = tf.keras.layers.concatenate([x.output, y.output, z.output])

  #Common network
  n = tf.keras.layers.Dense(256, activation='relu')(combined)
  n = tf.keras.layers.Dense(dataShape**2, activation='linear')(n)
  n = tf.keras.layers.ReLU()(n)
  n = tf.keras.layers.Lambda(lambda l: ApplyKernel(inputs[0], l))(n)

  #Model
  model = tf.keras.Model(inputs=[inputs[0], x.input, y.input, z.input], outputs=n, name=modelName)

  return model