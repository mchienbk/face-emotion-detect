# =========================================================================
# Smile detection CNN Trainning
# Filename:    cnn_train.py
# Name:         Tran Minh Chien
# Date:         2019.12.22
# =========================================================================


# Importing all necessary libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.models import save_model 

img_width, img_height = 48, 48
  
# train_data_dir = 'D://Workspace//Projects//resources//emoji-dataset//train'
# validation_data_dir = 'D://Workspace//Projects//resources//emoji-dataset//validation'
# nb_train_samples = 400 
# nb_validation_samples = 100
# epochs = 10
# batch_size = 16
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 


# Installed the CNN
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(7)) 
model.add(Activation('sigmoid')) 
# model.add(Flatten())
  
model.compile(loss ='binary_crossentropy', 
                     optimizer ='rmsprop', 
                   metrics =['accuracy']) 
  
train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                 shear_range = 0.2, 
                  zoom_range = 0.2, 
            horizontal_flip = True) 
  
test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
train_generator =train_datagen.flow_from_directory(
    'D:\Workspace\\Projects\\resources\\emoji-dataset\\train',
    target_size=(48, 48),
    batch_size=32,
    classes=('angry','disgust','fear','happy','neutral','sad','surprise'),
    class_mode='categorical'
)
  
validation_generator = test_datagen.flow_from_directory( 
    'D:\\Workspace\\Projects\\resources\\emoji-dataset\\validation',
    target_size=(48, 48),
    batch_size=32,
    classes=None,
    class_mode='categorical')
  
# model.fit_generator(train_generator, 
#     steps_per_epoch = nb_train_samples // batch_size, 
#     epochs = epochs, validation_data = validation_generator, 
#     validation_steps = nb_validation_samples // batch_size) 
model.fit_generator(
    train_generator,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=2000
)


model.save('model_c.h5')