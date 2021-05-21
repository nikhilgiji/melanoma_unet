from config import * 
import numpy as np
import tensorflow as tf 
from datetime import datetime 
from tensorflow import keras
from unet_model import *

train_images    = np.load('train_images.npy')
test_images    = np.load('test_images.npy')
valid_images   = np.load('valid_images.npy')

train_mask    = np.load('train_mask.npy')
test_mask    = np.load('test_mask.npy')
valid_mask   = np.load('valid_mask.npy')

train_mask    = np.expand_dims(train_mask, axis=3)
test_mask    = np.expand_dims(test_mask, axis=3)
valid_mask   = np.expand_dims(valid_mask, axis=3)

print('Dataset Loaded') 

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Check_Points,
                                                 save_weights_only=True,
                                                 verbose=1)


history = model.fit(train_images, train_mask,
              batch_size=Batch_Size,
              epochs=EPOCHS,
              shuffle=True,
              verbose=1,
              validation_data=(valid_images, valid_mask),
              callbacks=[tensorboard_callback, checkpoint_callback]) 

print("Average test loss: ", np.average(history.history['loss'])) 