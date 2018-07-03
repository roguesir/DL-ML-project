from PIL import Image
import os
import numpy as np

list = os.listdir("./resize_image")
print(list)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

for image in list:
    im=Image.open("./resize_image/"+image)
    out = rgb2gray(im)
    out = out.resize((48, 48))

    #out.show()
    out.save("./resize_image/000-1.jpg")

