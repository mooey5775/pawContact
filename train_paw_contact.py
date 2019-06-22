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
    ap.add_argument('modelname', help="name of trained model")
    ap.add_argument('-d', '--data-path', default='images',
                    help="path to image data (default: images)")
    ap.add_argument('-bs', '--batch-size', type=int, default=64,
                    help="batch size when training")
    ap.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                    help="base learning rate")
    ap.add_argument('--disable-class-weighting', action='store_true',
                    help="OPTIONAL: disable class weighting in loss function")
    args = vars(ap.parse_args())

    print("[INFO] loading initial data... (size 42)")
    sz = 42
    lr = args['learning_rate']
    data = get_data(sz, args['data_path'])
    data = data.resize(int(sz*1.3), 'tmp')

    print("[INFO] initializing model...")
    learn = ConvLearner.pretrained(arch, data, precompute=True)

    # Configure class weighting
    if not args['disable_class_weighting']:
        num_items = [len(os.listdir(os.path.join(args['data_path'], 'train', i))) for i in data.classes]
        class_weights = [max(num_items) / i for i in num_items]
        weighted_class = [0.25, 0.75, 0.4, 0.5, 1, 1.25]
        class_weights = [i * j for (i, j) in zip(class_weights, weighted_class)]

        def weighted_nll(input, target, size_average=True, ignore_index=-100, reduce=True):
            return F.nll_loss(input, target,
                              weight=torch.Tensor(class_weights).cuda(),
                              size_average=size_average,
                              ignore_index=ignore_index, reduce=reduce)
        
        learn.crit = weighted_nll

    print("[INFO] training...")
    learn.fit(lr, 2)
    learn.precompute = False
    learn.fit(lr, 3, cycle_len=1)
    lrs = np.array([lr / 9, lr / 3, lr])
    learn.unfreeze()
    learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)
    learn.save(str(sz) + '_' + args['modelname'])

    print("[INFO] increasing size to 84")
    sz = 84
    learn.set_data(get_data(sz, args['data_path']))
    learn.freeze()
    learn.fit(lr, 3, cycle_len=1)
    learn.unfreeze()
    learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)
    learn.save(str(sz) + '_' + args['modelname'])

    print("[INFO] increasing size to 168")
    sz = 168
    learn.set_data(get_data(sz, args['data_path']))
    learn.freeze()
    learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
    learn.unfreeze()
    learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
    learn.save(str(sz) + '_' + args['modelname'])