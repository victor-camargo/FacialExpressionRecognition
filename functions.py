import cv2
import imutils
import h5py
import numpy as np

"""
This function uses a previously trained Haar Cascade to detect faces and crop it
input: single image
returns: faces images list, faces regions
"""
def get_faces(image):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if(image.shape[-1] > 1):
        # Convert into grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_list = []
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faces_list.append(image[y:y+h,x:x+w])
    return faces_list, faces

"""
This function pre process images for transfer_learning+posmodel system
input: single image
return: processed image
"""
def crop_to_net(image):
    #32x32
    resized = cv2.resize(image, (32,32))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized

"""
This functions get all images and labels in a dataset file
input: dataset file path
"""
def load_nonsplit_dataset(dataset_path):
    archive = h5py.File(dataset_path, 'r')
    images = np.array(archive.get('images'))
    labels = np.array(archive.get('labels'))

    return images, labels

"""
This functions get all encoded images and labels in a file
input: enconded dataset file path
"""
def load_encoded_dataset(dataset_path):
    archive = h5py.File(dataset_path, 'r')
    images = np.array(archive.get('encoded'))
    labels = np.array(archive.get('labels'))

    return images, labels

