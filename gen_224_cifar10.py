import numpy as np
import cv2
from keras.datasets import cifar10
import scipy
import os
def main():

    save_dir = './cifar10'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.array([scipy.misc.imresize(img, (224, 224)) for img in x_train])
    x_test = np.array([scipy.misc.imresize(img, (224, 224)) for img in x_test])

    ## init dict for file names
    class_cnt_dict = dict()
    for i in range(0,10):
        class_cnt_dict[str(i)] = 0

    for idx,each in enumerate(x_train):
        resized_image = scipy.misc.imresize(each, (224, 224))

        # class folder
        label = str(y_train[idx])
        class_folder = os.path.join(save_dir,label)

        os.makedirs(class_folder,exist_ok=True)

        # write image to disk
        cv2.write(resized_image,os.path.join(class_folder,class_cnt_dict[label]))

        class_cnt_dict[label] += 1

if __name__ == '__main__':
    main()