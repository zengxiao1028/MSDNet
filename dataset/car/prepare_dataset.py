import os
from shutil import copyfile
import cv2
import scipy.io as sio
sio.loadmat

def main():

    set ='train'

    src_folder = '/storage/car/data/image'
    dst_folder = '/storage/car/%s' % set

    with open('/storage/car/data/train_test_split/classification/%s.txt' % set) as f:

        lines = f.readlines()
        print(len(lines))
        for idx,line in enumerate(lines):
                line = line[:-1] if line[-1] == '\n' else line
                items = line.split('/')
                image_path = os.path.join(src_folder, line)

                label = items[1]
                save_folder = os.path.join(dst_folder, label)
                os.makedirs(save_folder, exist_ok=True)

                image_path_2 = os.path.join(save_folder, items[1]+items[2]+items[3])

                img = cv2.imread(image_path)
                img = cv2.resize(img,(224,224))
                cv2.imwrite(image_path_2, img)
                #copyfile(image_path, image_path_2)

def main_2():
    mat = sio.loadmat('/storage/standford_car/devkit/cars_train_annos.mat')
    a = mat['annotations'][0][0].real
    print(len(mat))

if __name__ == '__main__':
    main_2()