from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input, Dense, Conv2D, Flatten
from functions import load_dataset, crop_to_net
import numpy as np
import cv2

expressions = ['Neutral', 'Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']


json_file = open('transfer_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("transfer_model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#score = loaded_model.evaluate(X_test, Y_test)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Device Opened\n")
else:
    print("Failed to open Device\n")
    exit(1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while (True):
    ret, frame = cap.read()
    frame_det = np.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_det, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('teste',frame_det)
    if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
        for n,(x, y, w, h) in enumerate(faces):
            
            img = cv2.resize(frame[y:y+h,x:x+w, :],(32,32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img.reshape(1,*img.shape))
            data = loaded_model.predict(img)
            print(expressions[np.argmax(data)])
cap.release()
cv2.destroyAllWindows()

