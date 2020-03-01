# =========================================================================
# Smile detection CNN TEST
# Filename:    cnn_test.py
# Name:         Tran Minh Chien
# Date:         2019.12.22
# =========================================================================


# Importing all necessary libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.models import load_model
img_width, img_height = 48, 48
import cv2

if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 


# Testing
model = load_model('model_c.h5')

input_img = cv2.imread('test/0.jpg')
# input_img = cv2.resize(input_img,(48,48))
pred = model.predict(input_img.reshape(1, 48, 48,3))
print(pred)

input_img = cv2.imread('test/1.jpg')
# input_img = cv2.resize(input_img,(48,48))
pred = model.predict(input_img.reshape(1, 48, 48,3))
print(pred)

input_img = cv2.imread('test/2.jpg')
# input_img = cv2.resize(input_img,(48,48))
pred = model.predict(input_img.reshape(1, 48, 48,3))
print(pred)

input_img = cv2.imread('test/3.jpg')
# input_img = cv2.resize(input_img,(48,48))
pred = model.predict(input_img.reshape(1, 48, 48,3))
print(pred)


input_img = cv2.imread('test/4.jpg')
# input_img = cv2.resize(input_img,(48,48))
pred = model.predict(input_img.reshape(1, 48, 48,3))
print(pred)

input_img = cv2.imread('test/5.jpg')
# input_img = cv2.resize(input_img,(48,48))
pred = model.predict(input_img.reshape(1, 48, 48,3))
print(pred)

input_img = cv2.imread('test/6.jpg')
# input_img = cv2.resize(input_img,(48,48))
pred = model.predict(input_img.reshape(1, 48, 48,3))
print(pred)