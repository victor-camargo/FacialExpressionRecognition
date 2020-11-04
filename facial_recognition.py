from tensorflow import keras
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Flatten,Dense, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization
from keras.preprocessing import image
from keras.regularizers import l2
from keras.initializers import glorot_normal
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD, RMSprop
from functions import load_nonsplit_dataset, load_encoded_dataset
from sklearn.model_selection import train_test_split

import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
System Architecture of this project

                    CK+ images            #Images Dimensions (32,32,3)
                        |
                        |
                        |
          ______________V_______________
         |                             |
         |          ResNet50           |
         |  (Weights are not trained)  |  # ResNet50 used weights used on ImageNet, the last 1000 neurons classification layer was removed
         |_____________________________| 
                        |
                        |                 # Encoded vector(2048,1,1) of CK+ images accordingly to ResNet50 "knownledge"
          ______________V_______________
         |                             |
         |          Pos Model          |
         |    (Weights are trained)    |  # Fully Connected Multilayer Feedforward Neural Network
         |_____________________________|  # The input layer shape needs to be (1, 1, 2048) and the last layer has 7 classification neurons
                        |
                        |
                        V
    7 Facial Expressions Probabilities Classifications
                        

Facial Expressions used on this project
0: Angry
1: Contempt
2: Disgust
3: Fear
4: Happy
5: Sad
6: Surprise



"""



def pos_model(input_shape = (1,1,2048),was_trained = True, filename = "model.json", weights = "model.h5"):
    """
    This function defines the pos model for (32,32,3) images shape
    inputs: pos model input shape(needs to match with previous model last layer), boolean state if model was trained
            trained model filename, trained model weights file
    returns: keras model
    """
    
    # If model was trained and you wan't to compile and use it, just pass argument was_trained as True
    if was_trained:
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights)
        print("Modelo carregado")
        loaded_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return loaded_model
    

    dropout_val = 0.25 # 0.25

    # proposed FCNN, here don't have the necessity to use L1 or L2 regularization because of the Dropout and BatchNormalization layers
    # All layers have dropout, only the last 3 that don't have dropout
    
    model_in = Input(shape=input_shape)
    x = Flatten()(model_in)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_val)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_val)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_val)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_val)(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_val)(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    
    '''
    Obsolete model created on first versions
    x = Flatten()(model_in)
    x = Dense(512, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2(kernel_val), bias_regularizer=l2(bias_val))(x)
    x = Dropout(dropout_val)(x)
    x = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2(kernel_val), bias_regularizer=l2(bias_val))(x)
    x = Dropout(dropout_val)(x)
    x = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2(kernel_val), bias_regularizer=l2(bias_val))(x)
    x = Dropout(dropout_val)(x)
    x = Dense(16, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2(kernel_val), bias_regularizer=l2(bias_val))(x)
    '''
    model_out = Dense(7, activation='softmax')(x)
    return Model(name= 'pos_model',inputs= model_in, outputs = model_out)


def Resnet50_transfer():
    """
    This function returns ResNet50 model without last layer and with weights trained for ImageNet challenge
    """
    
    # load ResNet50 model with ImageNet weights without classifier layers
    model = ResNet50(include_top=False,
                  weights = 'resnet50_weights_notop.h5',
                  input_tensor=Input(shape=(32,32, 3)))
    
    # Freeze all weights of model, we will not train it
    for layer in model.layers:
        layer.trainable = False
        
    # define new model
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    return model


def transfer_learning_model(pre_model, pos_model, filename = "transfer_model.json", weights = "transfer_model.h5"):
    """
    This function create a sequential single model concatenating a transfer model and a pos model, in this sequence, and then it saves to the disk
    input: pre model, pos model, filename to save model structure, weights filename to save weights of created model
    """
    
    merged_model = Sequential()
    merged_model.add(pre_model)
    merged_model.add(pos_model)
    model_json = merged_model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    merged_model.save_weights(weights)
    print("Saved model to disk")
    return merged_model


def encode_data(data, labels, model, file_name= "encoded.hdf5"):
    """
    This function uses a transfer learning model to encode data on the last layer,
    it saves time when training a neural network with a freezed weights transfer learning model
    because you don't need to encode the dataset every training.
    After encoding the batch, it then saves it to a .hdf5 file, making it faster to read and save on the memory.

    inputs: an image batch, image batch labels,transfer learning trained keras compiled model, and file name to save weights
    """

    model.summary()
    print(data.shape)

    print("Imagens processadas")
    outputs = model.predict(data, batch_size=128, verbose=1)
    print(outputs.shape)
    
    f = h5py.File(file_name, mode ='w')
    f.create_dataset("encoded", outputs.shape, np.float32)
    f["encoded"][...] = outputs
    f.create_dataset("labels", (outputs.shape[0],7), np.uint8)
    f["labels"][...] = labels
    f.close()
    print('Finalizado')


def predict_data(images, model):
    """
    This function receives a batch of images  and keras compiled model, and then it prints an array of predicted images
    inputs: batch of images, compiled model
    """
    
    expressions = ['Neutral', 'Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
    image= preprocess_input(images.reshape(1,*images.shape))
    pred = model.predict(images)
    print(expressions[np.argmax(pred)])
    

def train_model(images, labels, nums_of_training=10, filename="model.json", weightsfile= "model.h5"):
    """
    This functions trains n times random set and weights pos model using encoded images and labels, saves best weights and structure
    inputs: encoded dataset, labels, number of training sessions, filename for saving model, file name for saving weights
    """

    acuracias = [] # list of all accuracies of trainments
    for i in range(nums_of_training):
        
        print("treinamento: "+str(i+1))

        model = pos_model((1,1,2048),False) #Here I created my model with different weights for each session of training
        
        # spliting the dataset in training, validation and test sets, respectively in 80%, 10%, 10%
        X_train, X_test, Y_train, Y_test = train_test_split(images, labels, train_size=0.8, test_size=0.2, stratify = labels)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, train_size=0.5, test_size=0.5, stratify= Y_test)

        # Saving the splitted dataset after each session of training
        f = h5py.File('models/train_val_test_'+str(i)+'.hdf5', mode ='w')
        f.create_dataset("X_train", X_train.shape, np.float32)
        f["X_train"][...] = X_train
        f.create_dataset("Y_train", (Y_train.shape[0],7), np.uint8)
        f["Y_train"][...] = Y_train
        
        f.create_dataset("X_val", X_val.shape, np.float32)
        f["X_val"][...] = X_val
        f.create_dataset("Y_val", (Y_val.shape[0],7), np.uint8)
        f["Y_val"][...] = Y_val
        
        f.create_dataset("X_test", X_test.shape, np.float32)
        f["X_test"][...] = X_test
        f.create_dataset("Y_test", (Y_test.shape[0],7), np.uint8)
        f["Y_test"][...] = Y_test
        f.close()

        #Some callback functions used to make the model more accurate and with fast convergence
        early_stopping_monitor = EarlyStopping( monitor='val_accuracy', min_delta=0, patience=21, verbose=0, mode='max')
        checkpoint = ModelCheckpoint('models/weights_best_'+str(i)+'.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=3, verbose=1, min_delta=1e-4, mode='max')

        # compilation and training of model
        model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=200, validation_data=(X_val, Y_val), batch_size = 256, verbose=0, callbacks=[early_stopping_monitor, checkpoint, reduce_lr_loss])

        #plotting the validation accuracy and loss into /models dir
        fig, ax = plt.subplots(2, 1)
        fig.subplots_adjust(hspace=0.5)

        ax[0].plot(history.history['accuracy']) #row=0, col=0
        ax[0].plot(history.history['val_accuracy']) #row=0, col=0
        ax[0].set_title('Model Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_xlim(0)
        ax[0].legend(['train', 'val'], loc='lower right')

        ax[1].plot(history.history['loss']) #row=1, col=0
        ax[1].plot(history.history['val_loss']) #row=1, col=0
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_xlim(0)
        ax[1].legend(['train', 'val'], loc='upper right')
        plt.savefig('models/results_'+str(i)+".png")

        acuracias.append(model.evaluate(X_test, Y_test)) # append model evaluation to accuracies list

        # saving model structure
        model_json = model.to_json()
        with open("models/"+str(i)+'_'+filename, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

    with open('models/acuracias.txt', 'w') as f:
        print(acuracias, file = f) # saving accuracies to models/acuracias.txt file






#images, labels = load_nonsplit_dataset('ck_resnet_dataset.hdf5') #CK+ augmented dataset
#labels = to_categorical(labels) # One-Hot encoded labels
#encode_data(images, labels,resnet_model, file_name='ck_resnet_encoded.hdf5')
        
#model = pos_model((1,1,2048),False)
#model.summary()
        
#resnet_model = Resnet50_transfer()
#resnet_model.summary()


images, labels = load_encoded_dataset('ck_resnet_encoded.hdf5') #ImageNet ResNet50 last layer encoded CK+ augmented images
train_model(images, labels, filename="ck_resnet_model.json", weightsfile= "ck_resnet_model.h5") # Train only pos model with encoded images






