import imageio, os
import numpy as np
from matplotlib import pyplot as plt
from colour import Color as c

os.chdir("D:/Bachelor_resources/")
baseImage = imageio.imread("Capture1/Capture1_FinalImage_0839.png")
lossImage = imageio.imread("Renders/3Depth_K201_LossRender_0.png")
baseColor = imageio.imread("Capture1/Capture1_SceneColor_0839.png")
# variationImage = imageio.imread("Renders/3Depth_K193_BaseVariation_01.png")

def MapColor(value, gradient) :
    return gradient[value]

def GradientMap(image, gradient) :
    image = image.astype('uint8')
    return MapColor(image, gradient)

def RenderLoss(y_true, y_pred) :
  delta = np.mean(np.absolute(y_true - y_pred), axis=2)
  return delta

red = c("red")
green = c("green")
yellow = c("yellow")
colors = list(green.range_to(yellow,128)) +list(yellow.range_to(red,128))


gradient = np.zeros((256, 3))
for color in range(len(colors)) :
    gradient[color,0] = colors[color].rgb[0]
    gradient[color,1] = colors[color].rgb[1]
    gradient[color,2] = colors[color].rgb[2]

variationImage = RenderLoss(baseImage, baseColor)
maxVariation = np.amax(variationImage)
print("Max variation: {}".format(maxVariation))
variationImage /= (210/maxVariation)

lossMap = GradientMap(lossImage, gradient)
variationMap = GradientMap(variationImage, gradient)

imageio.imwrite("Illustrations/LossMap_0.png", (lossMap * 255).astype('uint8'))
imageio.imwrite("Illustrations/VariationMap_0.png", (variationMap * 255).astype('uint8'))

fig = plt.figure()

fig.add_subplot(2, 1, 1)
plt.imshow(lossMap)

fig.add_subplot(2, 1, 2)
plt.imshow(variationMap)
plt.show()