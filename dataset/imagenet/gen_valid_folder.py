import numpy as np
import os
import shutil

def main():
    with open('') as f:
        lines = f.readlines()
        entries = [line.split() for line in lines]
        dict = {int(entries[1])-1 : entries[0]}

    src_folder = ''
    dst_folder = ''
    val_imgs = sorted(src_folder)

    assert len(val_imgs) == 50000

    for idx, img in enumerate(val_imgs):
        dst_class_folder = os.path.join(dst_folder, dict[idx] )
        os.makedirs(dst_class_folder)
        shutil.copyfile( os.path.join(src_folder, img), os.path.join(dst_class_folder,img))



if __name__ == '__main__':
    main()