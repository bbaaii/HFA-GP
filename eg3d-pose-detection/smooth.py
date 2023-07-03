import os
import torch 
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d

from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat

import json

from scipy.ndimage import gaussian_filter1d


def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    im_path = sorted(im_path , key=lambda x:int(x.split('/')[-1].split('.')[0]))
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]
    return  lm_path


def read_data( lm_path):
    lms = []  
    # for i in range(10):
    for i in range(len(lm_path)):
        #im = Image.open(im_path).convert('RGB')
        # _, H = im.size
        if not os.path.isfile(lm_path[i]):
            continue
        lm = np.loadtxt(lm_path[i]).astype(np.float32)
        # print(lm)
        lms.append(lm)
    lms = np.array(lms)
    lms = gaussian_filter1d(lms, 2, 0)

    # for i in range(10):
        # print(lms[i])
    for i in range(len(lm_path)):
        if not os.path.isfile(lm_path[i]):
            continue
        np.savetxt(lm_path[i], lms[i])


def main(rank, opt, name='examples'):

    lm_path = get_data_path(name)
    read_data(lm_path)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt,opt.img_folder)
