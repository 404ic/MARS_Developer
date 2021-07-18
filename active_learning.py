from re import M
import multibox_detection.train_detect as train_detector
import multibox_detection.evaluate_detection as evaluate_detector
import hourglass_pose.train_pose as train_pose
import hourglass_pose.evaluate_pose as evaluate_pose
import hourglass_pose.config as config
from create_new_project import *

from glob import glob
import os

def create_AL_project(project_path):
    # makes project at project_path/active_learning 
    project_dir = os.path.join(project_path, 'active_learning')
    os.mkdir(project_dir)
    print('Directory at ' + project_dir + ' created')
    create_new_project(project_path, 'active_learning', download_demo_data=False, download_MARS_checkpoints=False)

def AL_train_pose(project):
    # Paths
    cfg_path = os.path.join(project, 'pose', 'config_train.yaml')
    log_dir = os.path.join(project, 'pose', 'top_log')
    # Parse configuration file
    cfg = config.parse_config_file(cfg_path)
    tf_records = os.path.join(project, 'pose', 'top_tfrecords_pose')
    # Actually run the training.
    train_pose.train(
        tfrecords_train=glob(os.path.join(tf_records, 'train_dataset-*')),
        tfrecords_val=glob(os.path.join(tf_records, 'val_dataset-*')),
        logdir=log_dir,
        cfg=cfg
        )

def AL_evaluate_pose(project, iteration):
    log = os.path.join(project, 'pose', 'top_log')
    ckpt_nums = glob.glob(os.path.join(log, 'model.ckpt-*'))
    ckpt_nums = [num for num in ckpt_nums if 'index' not in num and 'data' not in num]
    ckpt_nums = [int(num.split('-')[1].split('.')[0]) for num in ckpt_nums]
    ckpt_nums.sort()
    ckpt_nums = [ckpt_nums[-1]]
    for ckpt_num in ckpt_nums:
        print('============================================================')
        print('Processing model', ckpt_num)
        evaluate_pose.run_test(project, ckpt_num=ckpt_num, iteration=iteration)
        evaluate_pose.plot_model_PCK(project, ckpt_num=ckpt_num, iteration=iteration)
        evaluate_pose.plot_score_vs_accuracy(project=project, ckpt_num=ckpt_num, iteration=iteration)
        print('============================================================')
    return

def AL_train_iteration(project, iteration):
    AL_train_pose(project)
    AL_evaluate_pose(project, iteration)
    # need to add code to process evaluations

