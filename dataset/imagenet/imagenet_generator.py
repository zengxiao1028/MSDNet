from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import csv
#
# create image data generator from directory for imagenet
#
def data_generator(train_dir, val_dir, batch_size, top_n_classes=-1):

    classes = None
    if top_n_classes > 1:
        with open('dataset/imagenet/imagenet_classes.csv') as f:
            lines = f.readlines()[:top_n_classes]
            classes = [each.split(',')[0] for each in lines ]


    ### prepare dataset #####
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.1,
        horizontal_flip=True,
        rotation_range=30.,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes)

    return train_generator, validation_generator

if __name__ == '__main__':
    data_generator('','',5,10)
