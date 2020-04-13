#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input,Dense,Activation,BatchNormalization,Flatten,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import Sequence
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint 


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Check for dirs

train_dir = './data/train'
test_dir = './data/test'
model_dir = './models'

for dir in [train_dir,test_dir,model_dir]:
    if not(os.path.exists(dir)):
        os.mkdir(dir)


# In[4]:


# Loading info from parser

info = {
    "inputShape": (200,200,1),
    "autoencoderFile": os.path.join(model_dir, "autoecoder.h5"),
    "encoderFile": os.path.join(model_dir, "encoder.h5"),
    "decoderFile": os.path.join(model_dir, "decoder.h5"),
    "checkpointFile": os.path.join(model_dir, "checkpoint.h5"),
    "trainHistory": os.path.join(model_dir, "train_history.csv"),
    "mode": 'train',
    "retrain": True,
    "loss": 'binary_crossentropy',
    "optimizer": 'adam',
    "batchSize": 32,
    "multiprocessing": False
}


# In[5]:


# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
# flow_from_directory
# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

# ImageDataGenerator and flow_from_directory
# View documentation: https://keras.io/preprocessing/image/
train_datagen = image.ImageDataGenerator(rescale=1./255,
                                    validation_split=0.2
                                  )
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
            
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=info["inputShape"][:2],
    batch_size=info["batchSize"],
    color_mode='grayscale',
    class_mode='input',
    subset='training',
    seed=0
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=info["inputShape"][:2],
    batch_size=info["batchSize"],
    color_mode='grayscale',
    class_mode='input',
    subset='validation',
    seed=0
)


# In[6]:


validation_generator.samples//32


# In[7]:


class Autoencoder():
    
    def __init__(self, info):
        self.info= info

    
    def build_models(self):
        
        # Build and compile
        # Print building models
        conv_shape = (3,3) # convolutional kernel shape
        pool_shape = (2,2) # pooling kernel shape
        n_hidden_1, n_hidden_2, n_hidden_3 = 16, 8, 8 # channel numbers
        input_shape = self.info['inputShape']
        input_layer = Input(shape= input_shape)
        
        #encoder layers
        x = Conv2D(n_hidden_1, conv_shape, activation='relu', padding='same')(input_layer)
        x = MaxPooling2D(pool_shape, padding='same')(x)
        x = Conv2D(n_hidden_2, conv_shape, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_shape, padding='same')(x)
        x = Conv2D(n_hidden_3, conv_shape, activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pool_shape, padding='same')(x)
        
        #decoded layers
        x = Conv2D(n_hidden_3, conv_shape, activation='relu', padding='same')(encoded)
        x = UpSampling2D(pool_shape)(x)
        x = Conv2D(n_hidden_2, conv_shape, activation='relu', padding='same')(x)
        x = UpSampling2D(pool_shape)(x)
        x = Conv2D(n_hidden_1, conv_shape, activation='relu', padding='same')(x)
        x = UpSampling2D(pool_shape)(x)
        decoded = Conv2D(input_shape[2], conv_shape, activation='sigmoid', padding='same')(x)
        
        # Creating Autoencoder
        autoencoder = Model(input_layer,decoded)
        # Creating Encoder
        encoder = Model(input_layer,encoded)
        
        # Output encoder shapes
        output_encoder_shape = encoder.layers[-1].output_shape[1:]

        # Create decoder model (Reverse)
        decoded_input = Input(shape=output_encoder_shape)
        
        decoded_output = autoencoder.layers[-7](decoded_input)  # Conv2D
        decoded_output = autoencoder.layers[-6](decoded_output)  # UpSampling2D
        decoded_output = autoencoder.layers[-5](decoded_output)  # Conv2D
        decoded_output = autoencoder.layers[-4](decoded_output)  # UpSampling2D
        decoded_output = autoencoder.layers[-3](decoded_output)  # Conv2D
        decoded_output = autoencoder.layers[-2](decoded_output)  # UpSampling2D
        decoded_output = autoencoder.layers[-1](decoded_output)  # Conv2D
        
        decoder = Model(decoded_input, decoded_output)
        
        # Generate summaries
        print("\nautoencoder.summary():")
        print(autoencoder.summary())
        print("\nencoder.summary():")
        print(encoder.summary())
        print("\ndecoder.summary():")
        print(decoder.summary())
        
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        
        print('Models succesfully built')
        
        self.compile(loss = self.info['loss'], optimizer = self.info['optimizer'])
        
        print('Building and compilation done')
        
    def compile(self, loss='binary_crossentropy', optimizer='adam'):
        print('Compiling models...')
        # To fit the model using the parameters
        self.autoencoder.compile(loss=loss,optimizer=optimizer)
    
    def predict_embedding(self,X):
        return self.encoder.predict(X)
    
    def reconstruct_img(self,X):
        return self.autoencoder.predict(X)
    
    def fit(self, train_generator, validation_generator, n_epochs=2, batch_size=256, callbacks=[]): 
        # Split the train test set
        print('Fitting models....')

        self.autoencoder.fit_generator(
                    train_generator,
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data = validation_generator, 
                    validation_steps = validation_generator.samples // batch_size,
                    verbose=1,
                    epochs = n_epochs,
                    use_multiprocessing=self.info["multiprocessing"],
                    callbacks=callbacks,
        )
    
    def save_models(self):
        print('Saving models...')
        self.autoencoder.save(self.info["autoencoderFile"])
        self.encoder.save(self.info["encoderFile"])
        self.decoder.save(self.info["decoderFile"])
        
        print('models.saved')
        
    def load_models(self, loss='binary_crossentropy', optimizer='adam'):
        print('Loading and compiling models..')
        self.autoencoder = load_model(self.info["autoencoderFile"])
        self.encoder = load_model(self.info["encoderFile"])
        self.decoder = load_model(self.info["decoderFile"])
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.encoder.compile(optimizer=optimizer, loss=loss)
        self.decoder.compile(optimizer=optimizer, loss=loss)
        
        print('Loading and compiling models done')
        
        


# In[8]:


model = Autoencoder(info)


if info['retrain']:
    model.build_models()
else:
    if os.path.isfile(info['autoencoderFile']):
        model.load_models()
    else:
        model.build_models()


# In[9]:



train_history = info['trainHistory']
checkpoint_file = info['checkpointFile']
csv_logger=CSVLogger(train_history, append=True, separator=';')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=40)


# In[ ]:


# Train
if __name__ == "__main__":
    if info['mode']=='train':
        hist = model.fit(train_generator,
                         validation_generator, 
                         n_epochs=200,
                         batch_size=32,
                         callbacks=[csv_logger,es,checkpoint]
                 )
        model.save_models() # Save encoder models

    
    #csv_logger
    # early stopping


# In[ ]:





# In[ ]:




