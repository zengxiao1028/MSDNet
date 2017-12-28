import os
from shutil import copyfile


labels = {'(0, 2)':'0',
          '(4, 6)': '1',
          '(8, 13)': '2',
          '(15, 20)': '3',
          '(25, 32)': '4',
          '(38, 43)': '5',
          '(48, 53)': '6',
          '(60, 100)': '7',}
def main():

    scr_folder = ''
    dst_folder = ''
    with open('fold_0_data.txt') as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if idx ==0:
                continue
            else:
                items = line.split()
                image_path = items[0] + '/coarse_tilt_aligned_face.' + items[2] + '.' + items[1]
                label = labels[items[3]]

                save_folder = os.path.join(dst_folder,label)
                os.makedirs(save_folder, exist_ok=True)
                copyfile(os.path.join(scr_folder,image_path),os.path.join(save_folder, items[0]+'_' +items[2]))

if __name__ == '__main__':
    main()