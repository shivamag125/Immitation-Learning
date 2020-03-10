# victorylap code:
import numpy as np
import pandas as pd 
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Lambda,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import math
import json
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from time import time
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
#GPU imports
import tensorflow as tf
from keras import backend as k
from keras.callbacks import EarlyStopping
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################


def_path="/home/naruarjun/27th_testing/data_24/"
path_csv="/home/naruarjun/27th_testing/data_24/driving_log.csv"
data=pd.read_csv(path_csv, usecols=['center','left','right','steering','speed'], dtype={"steering":float, "speed":float})
#print(data)
l,n=data.shape 
data=data[data['speed']>25].reset_index()
#print(data.steering[0])

def img_aug_bright(img):
        #img=cv2.cvtColor(img,cv2.COLOR_RGB2RGB)
        fact=0.25 + np.random.uniform()
        img_hsv= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv[:,:,2]=img_hsv[:,:,2]*fact
        img_hsv[:,:,2][img_hsv[:,:,2]>255]=255
        img_rgb=cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img_rgb

def img_aug_shift(img, steer):
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        rows, cols,k=img.shape
        tr_x = 150*np.random.uniform()-75
        steer_ang = steer + tr_x*0.004
        tr_y = 10*np.random.uniform()-5
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        image_tr = cv2.warpAffine(img,Trans_M,(cols,rows))
        return image_tr, steer_ang

def img_shadow(img):
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image_shadow=img
        x1=np.random.randint(0,75)
        x2=np.random.randint(x1, 150)
        y1=np.random.randint(0,150)
        y2=np.random.randint(y1, 300)
        image_shadow[x1:x2, y1:y2]=0.3*image_shadow[x1:x2, y1:y2]
        image = cv2.addWeighted(img,0.5,image_shadow,0.5,0)
        return image
        
def img_flip(img, steer):
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.flip(img,1)
        steer=-1.0*steer
        return img, steer

new_size_col = 200
new_size_row = 66

def preprocessImage(image):
        shape = image.shape
        # note: numpy arrays are (row, col)!
        image = image[42:135]
        image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)    
        #image = image/255.-.5
        return image 



def pre_process(line_data):
        i_lrc= np.random.randint(3)

        if i_lrc == 0:
                path_file = line_data['center'][0].strip()
                shift_ang = 0
        if i_lrc == 1:
                path_file = line_data['left'][0].strip()
                shift_ang = 0.25
        if i_lrc == 2:
                path_file = line_data['right'][0].strip()
                shift_ang = -1*0.25
        y_steer = line_data['steering'][0] + shift_ang
        path_file=def_path+path_file
        image = cv2.imread(path_file)
        
        image, y_steer=img_aug_shift(image, y_steer)

        image=img_aug_bright(image)
        

        k=np.random.randint(2)
        if k==0:
                image, y_steer=img_flip(image, y_steer)
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image =cv2.equalizeHist(image)
        k=np.random.randint(2)
        if k==0:
                image=img_shadow(image)
        image=preprocessImage(image)
        
        return image, y_steer


def pre_process_test(line_data):
        path_file = line_data['center'][0].strip()
        #y_steer = line_data['steering'][0]
        path_file= def_path+ path_file
        image = cv2.imread(path_file)
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image=preprocessImage(image)
        #image =cv2.equalizeHist(image)
        return image

p_threshold=1
def train_gen(batch_size, data):
        batch_images = np.zeros((batch_size,64,64,1))
        batch_steering = np.zeros(batch_size)
        while True:
                for i in range(0, batch_size):
                        i_line= np.random.randint(len(data))
                        line_data= data.iloc[[i_line]].reset_index()
                        k=0
                        while k==0:
                                x,y=pre_process(line_data)
                                #cv2.imshow('1', x)
                                #cv2.waitKey(1000)
                                #print(y)
                                if abs(y)<0.25:
                                        p_val = np.random.uniform()
                                        if p_val>p_threshold:
                                                k = 1
                                else:
                                        k=1  
                        x=np.resize(x,(64,64,1))      
                        batch_images[i]=x
                        batch_steering[i]=y
                #plt.hist(batch_steering, normed=True, bins="auto")
                #plt.show()
                yield batch_images, batch_steering
def test_gen(data):
        while 1:
                for i_line in range(len(data)):
                        line_data = data.iloc[[i_line]].reset_index()
                        x = pre_process_test(data)
                        x = x.reshape(1, x.shape[0], x.shape[1],1)
                        y = line_data['steering'][0]
                        y = np.array([[y]])
                        yield x, y


#training_gen=train_gen(128, data)
val_gen=test_gen(data)
#print(type(val_gen))

model =Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(64,64,1)))

#Learns 3 best colour spaces
model.add(Conv2D(3,kernel_size=(1,1),  kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(5,5), strides=(2,2),  kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(5,5), strides=(2,2),  kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(32,kernel_size=(3,3), strides=(1,1),  kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(32,  kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(32,  kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(Dropout(0.5))


model.add(Dense(1, kernel_initializer='he_normal'))

model.compile(loss='mse',optimizer='adam', metrics=['mse'])
model.summary()
val_size=len(data)
early=EarlyStopping(monitor='val_loss',mode='auto', patience=2)
for i in range(0,80):
        training_gen=train_gen(32, data)
        model.fit_generator(training_gen, steps_per_epoch=160, epochs=100, validation_data=val_gen, validation_steps=val_size, callbacks=[early,tensorboard])
        model_name="model_iteration"+str(i)+".h5"
        model.save(model_name)
        if i==30:
            p_threshold=1
        else:
            p_threshold=1/(i+1)

        #model_json=model.to_json()
        #with open("model3.json","w") as f:
        #        json.dump(model_json,f)


        #model.save_weights("weights1.h5")



                






