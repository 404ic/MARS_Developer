from pose_annotation_tools.annotation_postprocessing import process_active_learning_data
from re import M
import multibox_detection.train_detect as train_detector
import multibox_detection.evaluate_detection as evaluate_detector
import new_eval as eval
import multibox_detection.prcurve_separate as prcurve
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

def AL_evaluate_detector(project, iteration):
    evaluate_detector.run_test(project)
    models = ['black_top']
    for model in models:
        evaluation_dir = os.path.join(project, 'detection', model + '_evaluation') # reference to specific mouse color
        evaluate_detector.pr_curve(project)
        os.rename(os.path.join(evaluation_dir, 'performance_detection.json'), os.path.join(evaluation_dir, 'performance_detection_iteration_' + str(iteration) + '.json'))
        os.rename(os.path.join(evaluation_dir, 'PR_curves.png'), os.path.join(evaluation_dir, 'pr_curves_iteration_' + str(iteration) + '.png'))

def AL_rename_detector(project, iteration):
    detection_dir = os.path.join(project, 'detection')
    os.rename(os.path.join(detection_dir, 'black_top_log'), os.path.join(detection_dir, 'black_top_log_iteration_' + str(iteration)))
    os.rename(os.path.join(detection_dir, 'black_top_tfrecords_detection'), os.path.join(detection_dir, 'black_top_tfrecords_detection_iteration_' + str(iteration)))


def AL_detector_choose_worst_images(project, iteration, num=10): # NEED TO CHANGE NUM=100 LATER
    infile = os.path.join(project, 'detection', 'black_top_evaluation', 'performance_detection_iteration_' + str(iteration) + '.json')
    with open(infile) as jsonfile:
        D = json.load(jsonfile)
    gt_keypoints = D['gt_bbox']['annotations']
    pred_keypoints = D['pred_bbox']
    data = []
    for i in range(len(gt_keypoints)):
        gt = gt_keypoints[i]
        pred = pred_keypoints[i*100]
        assert gt['image_id'] == pred['image_id'], 'Image IDs do not match'
        data.append([pred['image_id'],
                    pred['score']]
                    )
    data.sort(key=lambda pt: pt[1])
    data = np.array(data)
    worst_frames = [int(data[i][0]) for i in range(num)]
    return worst_frames

def AL_train_iteration_detection(project, iteration, frames_so_far, frames_to_add):
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
    train_detector.run_training(project)
    AL_evaluate_detector(project, iteration=iteration)
    worst_frames = AL_detector_choose_worst_images(project, iteration)
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
    AL_rename_detector(project, iteration)
    return frames_used_this_iteration, worst_frames

def run_AL_detection(project, num_iterations):
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
        frames_used, frames_to_add = AL_train_iteration_detection(project, iter, frames_so_far=frames_used, frames_to_add=frames_to_add)

if __name__ == '__main__':
    '''  
    After creating a project, you will need:
    1. Add images to /project/annotation_data/raw_images/
    2. Title your manifest file 'all_data.manifest' and put it in /project/annotation_data/
    I think that's it. 
    '''
    project = '/home/ericma/Documents/active_learning_detection_debug'
    run_AL_detection(project, 3)