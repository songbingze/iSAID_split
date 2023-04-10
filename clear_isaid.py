import cv2
import os
import numpy as np
from pathlib import Path
import argparse

def main(cfg):
    clear_folder = cfg.clear_folder
    clear_folder = Path(clear_folder)
    for file in clear_folder.iterdir():
        if '_RGB.png' not in str(file.name):

            file_name = file.stem
            file_name_seg = file_name + '_instance_color_RGB.png'
            file_name_seg = clear_folder / file_name_seg

            img = cv2.imread(str(file_name_seg))
            if np.sum(img) == 0:
                file_name = file.stem

                file_name_ins = file_name + '_instance_id_RGB.png'
                file_name_ins = clear_folder / file_name_ins
                
                os.remove(str(file))
                os.remove(str(file_name_seg))
                os.remove(str(file_name_ins))
                print('remove file: ', file_name, file_name_seg, file_name_ins)

def parse_args():
    parser = argparse.ArgumentParser(description='Clear iSAID dataset, delete the empty image')
    parser.add_argument('--clear-folder', help="path of directory need to be clear", default='./images', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)