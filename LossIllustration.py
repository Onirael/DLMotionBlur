import imageio, os
import numpy as np
from matplotlib import pyplot as plt
from colour import Color as c

os.chdir("C:/Bachelor_resources/")
baseImage = imageio.imread("Capture1_Sorted/FinalImage/Capture1_FinalImage_0839.png")[:,:,:3]
renderImage = imageio.imread("Renders/3Depth_K201_Render_00.png")[:,:,:3]
baseColor = imageio.imread("Capture1_Sorted/SceneColor/Capture1_SceneColor_0839.png")[:,:,:3]

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

renderLoss = RenderLoss(baseImage, renderImage)
maxLoss = np.amax(renderLoss)

print("Max variation: {}".format(maxVariation))
print("Max loss: {}".format(maxLoss))

variationImage /= (maxLoss/maxVariation)
renderLoss /= (maxLoss/maxVariation)

lossMap = GradientMap(renderLoss, gradient)
variationMap = GradientMap(variationImage, gradient)

# imageio.imwrite("Illustrations/LossMap_0.png", (lossMap * 255).astype('uint8'))
# imageio.imwrite("Illustrations/VariationMap_0.png", (variationMap * 255).astype('uint8'))

fig = plt.figure()

fig.add_subplot(2, 1, 1)
plt.imshow(lossMap)

fig.add_subplot(2, 1, 2)
plt.imshow(variationMap)
plt.show()