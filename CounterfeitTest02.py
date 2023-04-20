
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:25:21 2022

@author: ENGINEER
"""

#%%
#Initializing libraries 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import os
import pandas as pd

#%%
#Getting the working directory

cd = os.getcwd()
#%%
#Getting the working folders from directory
counterCD = cd+'\C'
FiveCD = cd+'\F'
TenCD = cd+'\T'

CounterFiles = []
FiveFiles = []
TenFiles = []

for files in os.listdir(counterCD) :
    CounterFiles.append(files)
    
for files in os.listdir(FiveCD) :
    FiveFiles.append(files)
    
for files in os.listdir(TenCD) :
    TenFiles.append(files)
    
#%%
#Getting the all frames from folders
DFCounter = pd.DataFrame(CounterFiles)
DFFive = pd.DataFrame(FiveFiles)
DFTen = pd.DataFrame(TenFiles)


DFCounter['y'] = 'CounterFeit'
DFFive['y'] = '500'
DFTen['y'] = '1000'

#%%
NewSampleCD = DFCounter.sample(frac = 1)
NewSampleCD[0] = counterCD +'\\'+ NewSampleCD[0]



#%%
#getting 70% frames for training and 15% for validation and testing 
CNewrows = len(NewSampleCD) * 0.7
CNewrows2 = CNewrows + (len(NewSampleCD)*0.15)
CNewrows3 = CNewrows2 + (len(NewSampleCD)*0.15)

CCDTraining = NewSampleCD[:int(CNewrows)]
CCDTesting  = NewSampleCD[int(CNewrows):int(CNewrows2)] 
CCDValidation  = NewSampleCD[int(CNewrows2):int(CNewrows3)]





#%%

NewSampleCD1 = DFFive.sample(frac = 1)
NewSampleCD1[0] = FiveCD +'\\'+ NewSampleCD1[0]
#%%
#getting 70% frames for training and 15% for validation and testing
FNewrows = len(NewSampleCD1) * 0.7
FNewrows2 = FNewrows + (len(NewSampleCD1)*0.15)
FNewrows3 = FNewrows2 + (len(NewSampleCD1)*0.15)

FCDTraining = NewSampleCD1[:int(FNewrows)]
FCDTesting  = NewSampleCD1[int(FNewrows):int(FNewrows2)] 
FCDValidation  = NewSampleCD1[int(FNewrows2):int(FNewrows3)]


#%%

NewSampleCD2 = DFTen.sample(frac = 1)
NewSampleCD2[0] = TenCD +'\\'+ NewSampleCD2[0]

#%%
#getting 70% frames for training and 15% for validation and testing
TNewrows = len(NewSampleCD2) * 0.7
TNewrows2 = TNewrows + (len(NewSampleCD2)*0.15)
TNewrows3 = TNewrows2 + (len(NewSampleCD2)*0.15)

TCDTraining = NewSampleCD2[:int(TNewrows)]
TCDTesting  = NewSampleCD2[int(TNewrows):int(TNewrows2)] 
TCDValidation  = NewSampleCD2[int(TNewrows2):int(TNewrows3)]



#%%
#getting input data
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
testing = ImageDataGenerator(rescale=1/255)

#%%
TrainingFull = pd.concat([pd.concat([TCDTraining,FCDTraining],axis=0),CCDTraining],axis=0)
ValidationFull = pd.concat([pd.concat([TCDValidation,FCDValidation],axis=0),CCDValidation],axis=0)
TestingFull = pd.concat([pd.concat([TCDTesting,FCDTesting],axis=0),CCDTesting],axis=0)

#%%
TestingFull.sample(frac=1)
TestingFull.sort_values(by = TestingFull[0],ascending=False)

#%%
TrainingFull['x'] = TrainingFull[0] 
ValidationFull['x'] = ValidationFull[0]
TestingFull['x'] = TestingFull[0]

#%%
#train_dataset = train.flow_from_directory(
 #   r'C:\Users\ENGINEER\Desktop\CNN-Counterfeit02\BaseData\Training', target_size=(54,96), batch_size=32, class_mode='categorical')
#validation_dataset = train.flow_from_directory(
  #  r'C:\Users\ENGINEER\Desktop\CNN-Counterfeit02\BaseData\Validation', target_size=(54,96), batch_size=32, class_mode='categorical')
#%%
#setting the data set parameters for training
train_dataset = train.flow_from_dataframe(
    dataframe=TrainingFull,
    directory=None,
    subset="training",
    x_col='x',
    y_col='y',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(54,96))
#%%
#setting the data set parameters for validation
validation_dataset = validation.flow_from_dataframe(
    dataframe=ValidationFull,
    directory=None,
    x_col='x',
    y_col='y',
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(54,96))

#%%
#setting the data set parameters for testing
test_dataset= testing.flow_from_dataframe(
    dataframe=TestingFull,
    directory=None,
    x_col='x',
    y_col='y',
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode=None,
    target_size=(54,96))

#%%
#setting the layers parameters for of model
train_dataset.class_indices
train_dataset.classes

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(54,96, 3)),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Conv2D(
                                        32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Conv2D(
                                        64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    
                                    tf.keras.layers.Flatten(),
                                    #
                                    tf.keras.layers.Dense(
                                        512, activation='relu'),
                                    #
                                    tf.keras.layers.Dense(
                                        3, activation='softmax')
                                    ])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=('accuracy'))

STEP_SIZE_TRAIN=train_dataset.n//train_dataset.batch_size
STEP_SIZE_VALID=validation_dataset.n//validation_dataset.batch_size
STEP_SIZE_TEST=test_dataset.n//test_dataset.batch_size

history = model.fit(train_dataset,
                    steps_per_epoch=5,
                    validation_data=validation_dataset,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

model.save('allLayermodel.h5')

#%%
# evaluating the model
model.evaluate(validation_dataset,
steps=STEP_SIZE_VALID)

#%%
STEP_SIZE_TEST=test_dataset.n//test_dataset.batch_size
test_dataset.reset()
pred=model.predict(test_dataset,
steps=STEP_SIZE_TEST,
verbose=1)

#%%
predicted_class_indices=np.argmax(pred,axis=1)

#%%
labels = (train_dataset.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#%%
#ploting the result of model
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch') 
plt.legend(['validation', 'test'], loc='upper left')
plt.show()

#%%
dir_path =r'C:\Users\ENGINEER\Desktop\CNN-Counterfeit03\96-54_size_Data\Testing'


#%%
ImagesList = []



#%%
TestingDirectory = list(TestingFull['x'])



ActualValue= list(TestingFull['y'])
#%%



ActualValue = [0 if ab =='1000' else 1 if ab == '500' else 2 for ab in ActualValue ]


#%%
PredictedValue = []

#%%
for single in TestingDirectory:
#%%


for i in TestingDirectory:
    img = image.load_img(i,target_size=(54,96))
    ImagesList.append(img)
    #plt.imshow(img)
    #plt.show()
    
    
    
#%%


for img in ImagesList:
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    val = model.predict(images)
    print(val)
    new_val = list(val[0])

    if new_val[0] == 1:
        PredictedValue.append(0)
        print("This is 1000")    
    elif new_val[1] == 1:
        PredictedValue.append(1)
        print("THis is 500")
     
    elif new_val[2] ==1:
        PredictedValue.append(2)
        print("This is a Fake Note")  
            
            

#%% testing ACCURACY
AccuracyList = []

for i in range(len(ActualValue)):
    if (ActualValue[i] == PredictedValue[i]):
        AccuracyList.append(1)
        


Accuracy_Score = (np.sum(AccuracyList) / len(ActualValue) ) * 100
print(Accuracy_Score)

    

    






