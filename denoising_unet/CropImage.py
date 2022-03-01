import torch
import numpy as np
import os
import skimage.io as io

root = "D:\\Dataset\\SIDD_Small_sRGB_Only\\Data\\"

std_height = 224
std_width = 224

GT_name = "GT_SRGB_010.PNG"
NOISY_name = "NOISY_SRGB_010.PNG"

data_names = os.listdir(root)
k = 0
for x in data_names:
    if not os.path.exists(root + "GroundTruth"):
        os.mkdir(root + "GroundTruth")
    if not os.path.exists(root + "Noisy"):
        os.mkdir(root + "Noisy")
    X = io.imread(root + x + "\\" + NOISY_name)
    Y = io.imread(root + x + "\\" + GT_name)

    m , n = X.shape[0] , X.shape[1]
    print("Processing : %s and %s" % (x + "/" + GT_name,x + "/" + NOISY_name))
    for i in range(m // std_height):
        for j in range(n // std_width):
            tempX = X[i * std_height : (i + 1) * std_height,j * std_width : (j + 1) * std_width,:]
            tempY = Y[i * std_height : (i + 1) * std_height,j * std_width : (j + 1) * std_width,:]
            io.imsave(root + "Noisy\\" + str(k) + ".png",tempX)
            io.imsave(root + "GroundTruth\\" + str(k) + ".png", tempY)
            k += 1
        tempX = X[i * std_height : (i + 1) * std_height,n - std_width : n,:]
        tempY = Y[i * std_height : (i + 1) * std_height,n - std_width : n,:]
        io.imsave(root + "Noisy\\" + str(k) + ".png", tempX)
        io.imsave(root + "GroundTruth\\" + str(k) + ".png", tempY)
        k += 1
    for j in range(n // std_width):
        tempX = X[m - std_height : m,j * std_width : (j + 1) * std_width,:]
        tempY = Y[m - std_height : m,j * std_width : (j + 1) * std_width,:]
        io.imsave(root + "Noisy\\" + str(k) + ".png", tempX)
        io.imsave(root + "GroundTruth\\" + str(k) + ".png", tempY)
        k += 1
