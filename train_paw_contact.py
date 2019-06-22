from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import os
import sys
import argparse

arch = resnext50

def get_data(sz, PATH, bs=64):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_basic, max_zoom=1.1)
    return ImageClassifierData.from_paths(PATH, bs=bs, tfms=tfms)

if __name__ == '__main__':
    if not os.path.exists(os.path.join('fastai', 'weights')):
        print("[ERROR] Manual prerun step required")
        print("Download this file: http://files.fast.ai/models/weights.tgz")
        print("And extract it in the 'fastai' directory")
        sys.exit()

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data-path', default='images',
                    help="path to image data (default: images)")
    ap.add_argument('-bs', '--batch-size', type=int, default=64,
                    help="batch size when training")
    args = vars(ap.parse_args())

