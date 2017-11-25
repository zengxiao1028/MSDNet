import numpy as np
import os
import shutil

def main():

    with open('/storage/imagenet/ILSVRC/devkit/data/map_clsloc.txt') as f:
        lines = f.readlines()
        entries = [line.split() for line in lines]
        dict = {int(entry[1]) : entry[0] for entry in entries}

    with open('/storage/imagenet/ILSVRC/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt') as f:
        lines = f.readlines()
        entries = [line for line in lines]
        dict2 = {}
        for idx,each in enumerate(entries):
            dict2[idx] = int(each)

    src_folder = '/storage/imagenet/ILSVRC/Data/CLS-LOC/val'
    dst_folder = '/storage/imagenet/ILSVRC/Data/CLS-LOC/val2'
    val_imgs = sorted(os.listdir(src_folder))

    print(len(val_imgs))
    assert len(val_imgs) == 50000

    for idx, img in enumerate(val_imgs):
        dst_class_folder = os.path.join(dst_folder, dict[dict2[idx]] )
        os.makedirs(dst_class_folder,exist_ok=True)
        shutil.copyfile( os.path.join(src_folder, img), os.path.join(dst_class_folder,img))



if __name__ == '__main__':
    main()