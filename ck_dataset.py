import h5py
import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet import preprocess_input
from random import shuffle
import matplotlib.pyplot as plt
'''

#Code created for generating data augmentation images

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))[:,:,::-1]
        if img is not None:
            images.append(img)
    return np.array(images)
for i in range(7):
    label = i
    path= "CK+/"+str(label)+"/"
    images = load_images_from_folder(path)
    labels = label*np.ones(images.shape[0], dtype=np.uint8)
    datagen = ImageDataGenerator(width_shift_range=2, height_shift_range = 2, rotation_range=30, horizontal_flip = True)
    datagen.fit(images)

    count = images.shape[0]
    for X_batch, y_batch in datagen.flow(images, labels, batch_size=1, save_to_dir=path, save_prefix='aug', save_format='png'):
        count = count + 1
        if(count >= 1500):
            break;

    
    #cv2.imshow('', X_batch.astype(np.uint8)[0])
    #cv2.waitKey(0)
    
'''




dirs = ["CK+/"+str(i) for i in range(7)]
faces_paths = []
labels = np.array([], dtype= np.uint8)
for count, i in enumerate(dirs):
    entire = [i +"/"+ k for k in os.listdir(i)]
    faces_paths= faces_paths+entire
    labels = np.concatenate((labels, count*np.ones(len(os.listdir(i)), dtype=np.uint8)), axis=None)
 


## Creating hdf5 file #########################################################
## All images are preprocessed before being encode by tranfer learning model###

file_name= 'ck_resnet_dataset.hdf5' # Created file for saving processed dataset
images_shape = (len(faces_paths), 32, 32, 3)

f = h5py.File(file_name, mode ='w')

f.create_dataset("images", images_shape, np.float32)


f.create_dataset("labels", (images_shape[0],), np.uint8)
f["labels"][...] = labels



for i in range(len(faces_paths)):

    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(faces_paths)) )

    addr = faces_paths[i]
    try:
        img = cv2.imread(addr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB
        img = cv2.resize(img, (32, 32))
        img = preprocess_input(img)
    except:
        print(addr)
    
    f["images"][i, ...] = img[None]

f.close()

