import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import preprocess_input
#from keras.applications.vgg16 import preprocess_input
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten,Dense
from keras import  Model
model = ResNet50(include_top=False,weights='imagenet',classes=431,input_shape=(224,224,3))
x = Flatten(name='flatten')(model.output)
x = Dense(431, activation='softmax', name='fc1')(x)
model = Model(model.input,x)
model.summary()
train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=False,
            rotation_range=15.,
            width_shift_range=0.1,
            height_shift_range=0.1)

train_generator = train_datagen.flow_from_directory(
    '../dataset/car/train/',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = test_datagen.flow_from_directory(
    '../dataset/car/test/',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical')

#### comppile model ########
opt = adam(lr=1e-4)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.samples // 64,
                                 epochs=100,
                                 validation_data=validation_generator,
                                 validation_steps=128,
                                 max_queue_size=32)