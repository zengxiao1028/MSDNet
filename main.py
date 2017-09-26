'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
import os
import pickle
import numpy as np
import MSDNet
import scipy
from numpy.random import shuffle
batch_size = 64
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'MSDNet_cifar10_trained_model_{}.h5'.format(time.time())
num_classifiers = 1

def data_generator(data_gen):

    while True:
        x, y = data_gen.next()
        yield x, [y] * num_classifiers


def prepro_fn(img):
    # img = scipy.misc.imresize(img, (224, 224))
    img = np.array(img).astype('float32')
    img = (img / 255. - 0.5) * 2.0
    return img

def main():

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')



    model = MSDNet.get_keras_model(x_train.shape)
    auto_save_callback = keras.callbacks.ModelCheckpoint(os.path.join('./models',model_name))
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')

        train_datagen = ImageDataGenerator(
            preprocessing_function=prepro_fn,
            shear_range=0.1,
            horizontal_flip=True,
            rotation_range=30.,
            width_shift_range=0.1,
            height_shift_range=0.1)

        test_datagen = ImageDataGenerator(preprocessing_function=prepro_fn)

        train_generator = train_datagen.flow_from_directory(
            './dataset/cifar10/train/',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            './dataset/cifar10/test/',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical')


        model.fit_generator(data_generator(train_generator),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data= data_generator(validation_generator),
                            validation_steps= x_test.shape[0] // batch_size,
                            callbacks=[auto_save_callback,tensorboard])

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Load label names to use in prediction results
    label_list_path = 'datasets/cifar-10-batches-py/batches.meta'



    keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
    datadir_base = os.path.expanduser(keras_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    label_list_path = os.path.join(datadir_base, label_list_path)

    with open(label_list_path, mode='rb') as f:
        labels = pickle.load(f)

    # Evaluate model with test data set and share sample prediction results
    evaluation = model.evaluate_generator(data_generator(validation_generator),
                                          steps=x_test.shape[0] // batch_size)

    print('Model Accuracy = %.2f' % (evaluation[1]))

    predict_gen = model.predict_generator(data_generator(validation_generator),
                                          steps=x_test.shape[0] // batch_size)

    for predict_index, predicted_y in enumerate(predict_gen):
        actual_label = labels['label_names'][np.argmax(y_test[predict_index])]
        predicted_label = labels['label_names'][np.argmax(predicted_y)]
        print('Actual Label = %s vs. Predicted Label = %s' % (actual_label,
                                                              predicted_label))
        if predict_index == num_predictions:
            break



if __name__ == '__main__':
    main()