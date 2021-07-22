import sys, os
sys.path.append('./evaluation')

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import yaml
import sys
import os
import glob
from multibox_detection.config import *


def plot_curve(path, log_path):
    save_name = path
    with open(log_path, 'rb') as fp: cocoEval = pickle.load(fp)
    print('loaded')

    #### PR CURVE
    rs_mat = cocoEval.params.recThrs #[0:.01:1] R=101 recall thresholds for evaluation
    ps_mat  = cocoEval.eval['precision']
    iou_mat = cocoEval.params.iouThrs
    arearng_mat = cocoEval.params.areaRng


    jet = cmx = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin = 0,vmax = len(iou_mat))
    scalarMap = cm.ScalarMappable(norm=cNorm,cmap= jet)

    show=[.5,.75,.85,.8999999999999999,.95]
    fig,ax = plt.subplots(1)
    for i in range(len(iou_mat)):
        if iou_mat[i] in show:
            colorVal = scalarMap.to_rgba(i)
            ax.plot(rs_mat,ps_mat[i,:,:,0,1],c=colorVal,ls='-',lw=2,label = 'IoU = %s' % np.round(iou_mat[i],2))
    plt.grid()
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc='best')
    # plt.legend(bbox_to_anchor=(1.1, 0.85),fontsize=12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.savefig(save_name)
    plt.show()


def prcurve(project, detector_names=[], max_training_steps=None, debug_output=False, log_path=None, save_path=None):
    # load project config
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not detector_names:
        detector_list = cfg['detection']
        detector_names = detector_list.keys()

    train_cfg = parse_config_file(os.path.join(project, 'detection', 'config_train.yaml'))

    # allow some command-line override of training epochs/batch size, for troubleshooting:
    if max_training_steps is not None:
        train_cfg.NUM_TRAIN_ITERATIONS = max_training_steps
    for detector in detector_names:
        logdir = os.path.join(project, 'detection', detector + '_log')
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        if log_path is None:
            log_path = os.path.join(project, 'detection', detector + '_tfrecords_detection', 'cocoEval.pkl')
        if save_path is None:
            save_path = os.path.join(project, 'detection')
        plot_curve(
            path=save_path,
            log_path=log_path
        ) 