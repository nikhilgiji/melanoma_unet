import os 
import glob 

Img_list = sorted(glob.glob('/path/to/melanoma/images/*.png')) 
Mask_list = sorted(glob.glob('/path/to/melanoma/mask/*.png')) 
#print(len(Img_list)) 
#print(len(Mask_list))

#model parameters 

IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3 

LR = 1e-4 
Batch_Size = 64
EPOCHS = 30

Check_Points = "/home/nikhil/tensorflow_dir/melanoma_unet/check_points/" 
