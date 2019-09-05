import tensorflow as tf
import numpy as np

def RGBtoXYZ(color):

    comp = tf.where(color > 0.04045, tf.ones(tf.shape(color)), tf.zeros(tf.shape(color)))
    newColor = comp * tf.math.pow((color + 0.055)/1.055, 2.4) + (1-comp) * color/12.92
    newColor *= 100

    convMat = tf.constant([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
        ])

    XYZ = tf.einsum('kj,ij->ik', convMat, newColor) # Possible rounding of result

    return XYZ

def XYZtoLAB(XYZ):
    newXYZ = XYZ / tf.constant([95.047, 100.0, 108.883])

    comp = tf.where(newXYZ > 0.008856, tf.ones(tf.shape(newXYZ)), tf.zeros(tf.shape(newXYZ)))
    newXYZ = comp * tf.math.pow(newXYZ, 1/3.0) + (1-comp) * (7.787 * newXYZ + 16/116.0)

    convMat = tf.constant([
        [0.0, 116.0, 0.0],
        [500.0, -500.0, 0.0],
        [0.0, 200.0, -200.0]
    ])

    Lab = tf.einsum('kj,ij->ik', convMat, newXYZ) + tf.constant([-16.0, 0.0, 0.0])

    return Lab

with tf.compat.v1.Session() as sess:
    k_pred = tf.random.uniform((5,3), minval= -0.1, maxval=1.1)
    # k_pred = tf.constant(
    # [[0.43492913, 0.28644,    0.02103555],
    #  [0.5158148,  0.26194036, 0.76836],
    #  [0.4722246,  0.6793847, 0.9088831],
    #  [0.5186044,  0.87784266, 0.22567332],
    #  [1.23, 0.9, 0.3],
    #  [1,1,1]]
    # )
    y_true = tf.random.uniform((5,3), minval = -0.1, maxval=1.1)
    # y_true = tf.constant(
    # [[0.8709409, 0.27560663, 0.11408544],
    # [0.12394309, 0.9468801, 0.36348748],
    # [0.97442305, 0.78993106, 0.5581573],
    # [0.08199382, 0.9744395, 0.82567394],
    # [0,0,0]]
    # )
    k_pred = sess.run(k_pred)
    y_true = sess.run(y_true)


    comp = tf.where(tf.math.logical_or(k_pred > 1.0, k_pred < 0.0), tf.ones(tf.shape(k_pred)), tf.zeros(tf.shape(k_pred)))
    comp2 = tf.reduce_max(comp, axis=1)

    expected = XYZtoLAB(RGBtoXYZ(y_true))

    invColorLoss = tf.reduce_sum(comp * (tf.where(k_pred > 1, k_pred - 1, tf.zeros(tf.shape(k_pred))) + \
        tf.where(k_pred < 0, -k_pred, tf.zeros(tf.shape(k_pred)))), axis=1) * 255.0 + 255.0 # The loss value for invalid colors (if RGB values are not in range 0 to 1)

    colorLoss = tf.norm(tf.abs(expected - XYZtoLAB(RGBtoXYZ(k_pred))), axis=1) # The loss value for valid colors
    colorLoss = tf.where(tf.math.is_nan(colorLoss), tf.zeros(tf.shape(colorLoss)), colorLoss)


    print("Expected :")
    print(expected.eval())
    print("Comparison 1 :")
    print(comp.eval())
    print("Comparison 2 :")
    print(comp2.eval())
    print("In prediction :")
    print(np.around(k_pred,2) * 255.0)
    print("Invalid color loss :")
    print(invColorLoss.eval())
    print("Valid color loss :")
    print(colorLoss.eval())
    print("Result :")
    result = comp2 * invColorLoss + (1-comp2) * colorLoss
    print(result.eval())


    k_pred = XYZtoLAB(RGBtoXYZ(k_pred))

    # print("Conversion to Lab :")
    # print(k_pred.eval())
    # print(expected.eval())

    # delta = tf.norm(tf.abs(expected - k_pred), axis=1)

    # print("Result :")
    # print(delta.eval())