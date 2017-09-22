import numpy as np
import cv2
from keras.datasets import cifar10
import scipy
import os
def main():

    save_dir = './dataset/cifar10/test'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()



    ## init dict for file names
    class_cnt_dict = dict()
    for i in range(0,10):
        class_cnt_dict[str(i)] = 0

    for idx,each in enumerate(x_test):
        resized_image = scipy.misc.imresize(each, (224, 224))

        # class folder
        label = str(y_test[idx][0])
        class_folder = os.path.join(save_dir,label)

        os.makedirs(class_folder,exist_ok=True)

        # write image to disk
        resized_image = resized_image[...,::-1]
        cv2.imwrite(os.path.join(class_folder, str(class_cnt_dict[label]) + '.jpg'),resized_image)

        class_cnt_dict[label] += 1

if __name__ == '__main__':
    main()