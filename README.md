# llSPS-INT-3743-Pneumonia-Prediction-using-X-RAY-images
Pneumonia Prediction using X-RAY images
A Convolutional Neural Network that is able to detect whether a patient has pneumonia, both bacterial and viral, or not, based on an X-ray image of their chest. Implements transfer learning, using the first 16 layers of a pre-trained VGG19 Network, to identify the image classes. The final accuracy obtained by the model, after testing on 624 unseen instances, is approximately 92%.

Execution Instructions:

1) first set ur environment in anaconda navigator 

create the pyhton version as 3.6

2) open anaconda prompt 

activate envipython (your environment name )

pip install tensorflow==1.14

3) after installing the tensorflow install keras

pip install keras==2.2.4

4)after installing keras install pandas

pip install pandas and same for other module ....!!

5)open jupyter notebook 

jupyter notebook

6)import these modules 

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

7) train the model 

model.fit_generator(x_train,samples_per_epoch=8000/32,epochs=100,validation_data=x_test,validation_steps=64) (you can use this or the code is given in the floder you can download the file 

FOR DATASET LINKS

Dataset Name     : Chest X-Ray Images (Pneumonia)
Dataset Link     : Chest X-Ray Images (Pneumonia) Dataset (Kaggle)
                 : Chest X-Ray Images (Pneumonia) Dataset (Original Dataset)
                 
Dataset Details
Dataset Name            : Chest X-Ray Images (Pneumonia)
Number of Class         : 2
Number/Size of Images   : Total      : 5856 (1.15 Gigabyte (GB))
                          Training   : 5216 (1.07 Gigabyte (GB))
                          Validation : 320  (42.8 Megabyte (MB))
                          Testing    : 320  (35.4 Megabyte (MB))

Model Parameters
Machine Learning Library: Keras
Base Model              : InceptionV3 && Custom Deep Convolutional Neural Network
Optimizers              : Adam
Loss Function           : categorical_crossentropy

For Custom Deep Convolutional Neural Network : 
Training Parameters
Batch Size              : 64
Number of Epochs        : 30
Training Time           : 2 Hours

Output (Prediction/ Recognition / Classification Metrics)
Testing
Accuracy (F-1) Score    : 89.53%
Loss                    : 0.41
Precision               : 88.37%
Recall (Pneumonia)      : 95.48%                

