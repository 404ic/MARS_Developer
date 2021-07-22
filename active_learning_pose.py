from pose_annotation_tools.annotation_postprocessing import process_active_learning_data
from re import M
import multibox_detection.train_detect as train_detector
import multibox_detection.evaluate_detection as evaluate_detector
import hourglass_pose.train_pose as train_pose
import hourglass_pose.evaluate_pose as evaluate_pose
import hourglass_pose.config as config
from create_new_project import *

from glob import glob
import os
import json
import numpy as np

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
    ckpt_nums = glob(os.path.join(log, 'model.ckpt-*'))
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

def AL_rename_pose(project, iteration):
    pose_dir = os.path.join(project, 'pose')
    os.rename(os.path.join(pose_dir, 'top_log'), os.path.join(pose_dir, 'top_log_iteration_' + str(iteration)))
    os.rename(os.path.join(pose_dir, 'top_tfrecords_pose'), os.path.join(pose_dir, 'top_tfrecords_pose_iteration_' + str(iteration)))

def AL_pose_choose_worst_images(project, iteration, num=100):
    file = os.path.join(project, 'pose', 'top_evaluation', '*_performance_pose_iteration_' + str(iteration) + '.json')
    file = glob(file)[0]
    with open(file) as model:
        data = json.load(model)
        images = data['gt_keypoints']['images']
        images = [x['id'] for x in images]
        pred_keypoints = data['pred_keypoints']
        avg_by_mouse = [x for x in pred_keypoints if x['category_id'] == 1]
        avg_by_frame = []
        for id in images:
            frames_for_given_id = [x for x in avg_by_mouse if x['image_id'] == id]
            avg_by_frame.append([id, np.mean([x['score'] for x in frames_for_given_id])])
        avg_by_frame.sort(key=lambda x: x[1])
        worst_frames = [avg_by_frame[i][0] for i in range(num)]
        # print(np.array(avg_by_frame))
        # print(worst_frames)
        # print(len(avg_by_frame))
        # print(len(images))
        return worst_frames

def AL_train_iteration_pose(project, iteration, frames_so_far, frames_to_add):
    '''
    Iteration 0: Randomly sample k images and train with those
    Iteration i: Use images chosen from iteration i-1 and train
    Returns frames used to train on, and then worst frames after this training iteration
    '''
    frames_used_this_iteration = []
    if iteration == 0:
        frames_used_this_iteration = process_active_learning_data(project, frames_included=[], frames_to_add=[], iteration=0, init_batch=100)
    else:
        frames_used_this_iteration = process_active_learning_data(project, frames_included=frames_so_far, frames_to_add=frames_to_add, iteration=iteration)
    print('Beginning training for iteration', iteration)
    AL_train_pose(project)
    AL_evaluate_pose(project, iteration)
    worst_frames = AL_pose_choose_worst_images(project, iteration)
    worst_frames = [int(x) for x in worst_frames]
    print(worst_frames)
    log = []
    if iteration != 0:
        with open(os.path.join(project, 'al_log.o'), 'r') as al_log:
            log = json.load(al_log)
            al_log.close()
    with open(os.path.join(project, 'al_log.o'), 'w') as al_log:
        dict = {}
        dict['iteration'] = iteration
        dict['training_frames'] = frames_used_this_iteration
        dict['most_uncertain_frames'] = worst_frames
        log.append(dict)
        json.dump(log, al_log)
        print('Iteration ' + str(iteration) + ' frames used to train: ')
        print(frames_used_this_iteration)
        print('Iteration ' + str(iteration) + ' worst frames: ')
        print(worst_frames)
        al_log.close()
    AL_rename_pose(project, iteration)
    return frames_used_this_iteration, worst_frames
    # need to add code to process evaluations

def run_AL_pose(project, num_iterations):
    print('Staring active learning cycles...')
    frames_used, frames_to_add = [], []
    start_iter = 0
    if os.path.isfile(os.path.join(project, 'al_log.o')):
        input = open(os.path.join(project, 'al_log.o'), 'r')
        data = json.load(input)
        start_iter = data[-1]['iteration'] + 1
        frames_used = data[-1]['training_frames']
        frames_to_add = data[-1]['most_uncertain_frames']
    for iter in range(start_iter, num_iterations):
        print('Starting iteration', iter)
        frames_used, frames_to_add = AL_train_iteration_pose(project, iter, frames_so_far=frames_used, frames_to_add=frames_to_add)

if __name__ == '__main__':
    '''
    After creating a project, you will need:
    1. Add images to /project/annotation_data/raw_images/
    2. Title your manifest file 'all_data.manifest' and put it in /project/annotation_data/
    I think that's it. 
    '''
    project = '/home/ericma/Documents/active_learning'
    run_AL_pose(project, 5)