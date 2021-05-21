import numpy as np 
import matplotlib.pyplot as plt
from config import *
from unet_model import *

test_images    = np.load('test_images.npy') 
test_mask    = np.load('test_mask.npy') 

plt.figure(figsize=(20, 9))
for i in range(10):
    plt.subplot(2,5,i+1) 
    plt.imshow(test_images[i], cmap="jet") 
plt.show()

plt.figure(figsize=(20, 9))
for i in range(10):
    plt.subplot(2,5,i+1) 
    predict_input = test_images[i]
    predictions =model.predict(predict_input.reshape(1,64,64,3), batch_size=1)
    prediction = predictions.reshape(64, 64) 
    plt.imshow(prediction, cmap="binary") 
plt.show()
