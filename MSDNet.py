import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D,Input, GlobalAveragePooling2D
from keras.models import Model
import config
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf

def get_model(x_train_shape):

    input = Input(shape=x_train_shape[1:] )

    conv1 = Conv2D(32, (3,3), activation='relu',padding='same')(input)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu',padding='same')(pool_1)
    conv4 = Conv2D(128, (3, 3), activation='relu')(conv3)

    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu',padding='same')(pool_2)

    conv6_flatten = Flatten()(conv5)

    fc1 = Dense(512,activation='relu')(conv6_flatten)
    fc1_dropout = Dropout(0.5)(fc1)



    ## generate prediction1
    c1 = Conv2D(32, (3, 3), activation='relu',padding='same')(conv2)
    c1 = Flatten()(c1)
    prediction1 = Dense(config.num_classes,name='c1',activation='softmax')(c1)

    ## generate prediction2
    c2 = Conv2D(32, (3, 3), activation='relu')(conv4)
    c2 = Flatten()(c2)
    prediction2 = Dense(config.num_classes,name='c2',activation='softmax')(c2)

    ##generate prediction3
    prediction3 = Dense(config.num_classes,activation='softmax',name='c3')(fc1_dropout)

    model = Model(inputs=input,outputs=[prediction1,prediction2,prediction3])

    # input = Input(shape=x_train_shape[1:])
    # net = Conv2D(32, (3, 3), padding='same',activation='relu')(input)
    # net = Conv2D(32, (3, 3), activation='relu')(net)
    # net = MaxPooling2D(pool_size=(2, 2))(net)
    # net = Dropout(0.25)(net)
    # net = Conv2D(64, (3, 3), padding='same',activation='relu')(net)
    # net = Conv2D(64, (3, 3), activation='relu')(net)
    # net = MaxPooling2D(pool_size=(2, 2))(net)
    # net = Dropout(0.25)(net)
    # net = Flatten()(net)
    # net = Dense(512,activation='relu')(net)
    # net = Dropout(0.5)(net)
    # net = Dense(config.num_classes)(net)
    #
    # model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  loss_weights=[0.5, 0.5, 0.5],
                  metrics=['accuracy'])

    return model


def get_sub_prediction(intermediate_output):
    alpha = 1.0
    shape = (1, 1, int(1024 * alpha))
    dp_rate = 0.5
    num_classes = 10

    x = GlobalAveragePooling2D()(intermediate_output)

    x = Reshape((1,1,x.get_shape().as_list()[1]))(x)
    x = Dropout(dp_rate)(x)
    x = Conv2D(num_classes, (1, 1),
               padding='same')(x)
    x = Activation('softmax')(x)
    x = Reshape((num_classes,))(x)
    return x

def get_keras_model(x_train_shape):

    model = MobileNet(weights='imagenet')

    intermediate_1 = get_sub_prediction(model.get_layer('conv_pw_1_relu').output)

    intermediate_2 = get_sub_prediction(model.get_layer('conv_pw_3_relu').output)

    intermediate_3 = get_sub_prediction(model.get_layer('conv_pw_5_relu').output)

    intermediate_4 = get_sub_prediction(model.get_layer('conv_pw_7_relu').output)

    intermediate_5 = get_sub_prediction(model.get_layer('conv_pw_9_relu').output)

    intermediate_6 = get_sub_prediction(model.get_layer('conv_pw_13_relu').output)


    # new_model = Model(inputs=model.input, outputs=[intermediate_1,intermediate_2,intermediate_3,intermediate_4,intermediate_5, intermediate_6])
    #
    # new_model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               loss_weights=[0.2] * 6,
    #               metrics=['accuracy'])


    new_model = Model(inputs=model.input,
                      outputs=intermediate_6)

    new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    return new_model

if __name__ == '__main__':
    get_keras_model(None)