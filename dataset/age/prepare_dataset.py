import os
from shutil import copyfile


labeks = {'(0, 2)':'0',
          ''(25, 32)':'}
def main():

    with open('fold_0_data.txt') as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if idx ==0:
                continue
            else:
                items = line.split()
                image_path = items[0] + '/coarse_tilt_aligned_face.' + items[2] + '.' + items[1]

                if


if __name__ == '__main__':
    main()