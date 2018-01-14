import os
from shutil import copyfile


labels = {'(0, 2)':'0',
          '(4, 6)': '1',
          '(8, 12)': '2',
          '(15, 20)': '3',
          '(25, 32)': '4',
          '(38, 43)': '5',
          '(48, 53)': '6',
          '(60, 100)': '7',}
def main():

    scr_folder = '/storage/faceage/faces'
    dst_folder = '/storage/faceage/faces3/train'

    with open('fold_4_data.txt') as f:
        lines = f.readlines()
        print(len(lines))
        for idx,line in enumerate(lines):
            if idx ==0:
                continue
            else:
                items = line.split('\t')
                image_path = items[0] + '/coarse_tilt_aligned_face.' + items[2] + '.' + items[1]
                # if items[3] in labels.keys():
                #     label = labels[items[3]]
                # else:
                #     print('no label')
                #     continue
                label = items[4]
                if label not in ['u','f','m']:
                    print('error')
                save_folder = os.path.join(dst_folder,label)
                os.makedirs(save_folder, exist_ok=True)
                #copyfile(os.path.join(scr_folder,image_path),os.path.join(save_folder, items[0]+'_' +items[1]))

if __name__ == '__main__':
    main()