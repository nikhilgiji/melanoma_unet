import os
import numpy as np 
from config import *
from skimage.io import imread
from skimage.transform import resize 
import tensorflow as tf 

Img_train   = np.zeros([2594, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
Mask_train  = np.zeros([2594, IMG_HEIGHT, IMG_WIDTH]) 

print('Train Images Reading....')
for idx in range(len(Img_list)):
    img = imread(Img_list[idx])
    img = np.double(resize(img, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], anti_aliasing=True))
    Img_train[idx, :,:,:] = img
print('Train Images Reading Finished')

print('Train Mask Reading....')
for idl in range(len(Mask_list)):
    img2 = imread(Mask_list[idl])
    img2 = np.double(resize(img2, [IMG_HEIGHT, IMG_WIDTH], anti_aliasing=True))
    Mask_train[idl, :,:] = img2
print('Train Mask Reading Finished') 

Train_Img      = Img_train[0:1815,:,:,:]
Validation_Img = Img_train[1815:1815+259,:,:,:]
Test_Img       = Img_train[1815+259:2594,:,:,:]

Train_Mask      = Mask_train[0:1815,:,:]
Validation_Mask = Mask_train[1815:1815+259,:,:]
Test_Mask       = Mask_train[1815+259:2594,:,:]

#print("shape of train image data is", tf.shape(Train_img))
#print("shape of valid image data is", tf.shape(Validation_img))
#print("shape of train test data is",tf.shape(Test_img)) 
#print("shape of train mask data is", tf.shape(Train_mask))
#print("shape of valid mask data is", tf.shape(Validation_mask))
#print("shape of train mask data is",tf.shape(Test_mask)) 

#save images and masks as numpy arrays 
np.save('train_images', Train_Img)
np.save('test_images' , Test_Img)
np.save('valid_images'  , Validation_Img)

np.save('train_mask', Train_Mask)
np.save('test_mask' , Test_Mask)
np.save('valid_mask'  , Validation_Mask) 

print("saved images and masks")