import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import cv2
from keras.datasets import cifar10
import scipy
import os
from keras.layers import Dense
from sklearn.externals import joblib
from keras.models import load_model
from keras.models import Model
def main(model):
    save_dir = './dataset/cifar10/train_teacher'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    y_train_soft = []
    for idx, each in enumerate(x_train):
        print(idx)
        resized_image = scipy.misc.imresize(each, (224, 224))
        resized_image = resized_image[np.newaxis, ...]
        prediction = model.predict(resized_image)

        y_train_soft.append(prediction[0])

    y_train_soft = np.array(y_train_soft)
    joblib.dump((x_train, y_train, y_train_soft), 'cifar10_train.pkl')

    y_test_soft = []
    for idx, each in enumerate(x_test):
        print(idx)
        resized_image = scipy.misc.imresize(each, (224, 224))
        resized_image = resized_image[np.newaxis, ...]
        prediction = model.predict(resized_image)

        y_test_soft.append(prediction)

    y_test_soft = np.array(y_test_soft)
    joblib.dump((x_test, y_test, y_test_soft), 'cifar10_test.pkl')

    x_test, y_test, y_test_soft = joblib.load('cifar10_test.pkl')

if __name__ == '__main__':
    model = load_model('/home/biyi/PycharmProjects/MSDNet/models/VGG16-E40p-9364-1fc512-10epoch.h5')

    for layer in model.layers:
        print(layer.name)
    fc1_layer_weights = model.get_layer('fc1').get_weights()

    new_fc1_layer = Dense(fc1_layer_weights[0].shape[1], activation=None, name='new_fc1')


    new_fc1_output = new_fc1_layer(model.get_layer('flatten').output)
    new_fc1_layer.set_weights(fc1_layer_weights)

    new_model = Model(model.input, new_fc1_output)

    main(new_model)
