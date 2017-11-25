import os
import webbrowser

def open_tabs(classes_list):

    wnids = [each.split()[0] for each in classes_list]

    ks = list(range(0,500,50))          # [0,50,100,150,200,250,300,350,400,450]

    k = ks[9] # change this from 0 to 9, every time open 50 tabs

    for i in range(k, k + 50):
       webbrowser.open_new_tab('http://image-net.org/synset?wnid=%s' % wnids[i])




def main(label_path):
    with open(label_path) as f:
        lines = f.readlines()

    classes_list_part1 = lines[:500]
    print('xiao handle',len(classes_list_part1))


    classes_list_part2 = lines[500:]
    print('biyi handle',len(classes_list_part2))


    open_tabs(classes_list_part1)  # change classes_list_part1 to classes_list_part2





if __name__ == '__main__':
    main('./dataset/imagenet/map_clsloc.txt')
