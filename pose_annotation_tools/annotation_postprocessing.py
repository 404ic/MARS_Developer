from hourglass_pose.train_pose import train
import pose_annotation_tools.tfrecord_util
import importlib
importlib.reload(pose_annotation_tools.tfrecord_util)
from pose_annotation_tools.tfrecord_util import *
import pose_annotation_tools.json_util
importlib.reload(pose_annotation_tools.json_util)
from pose_annotation_tools.json_util import *
from pose_annotation_tools.priors_generator import *
import random
import yaml
import json
import os
import glob
import argparse
import shutil

from pprint import pprint


def make_clean_dir(output_dir):
    if os.path.exists(output_dir):
        # remove all old tfrecords and priors for safety
        oldrecords = []
        for filetype in ['tfrecord','prior']:
            oldrecords.extend(glob.glob(os.path.join(output_dir, '*' + filetype + '*')))
        for f in oldrecords:
            try:
                os.remove(f)
            except OSError as e:
                print('Error: %s : %s' % (f, e.strerror))

    else:
        os.makedirs(output_dir)


def prepare_detector_training_data(project):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    and prior files that are used to train MARS's detectors and pose estimators for black and white mice.
    """
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    detector_list = config['detection']
    detector_names = detector_list.keys()

    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)
    # random.shuffle(D)

    for detector in detector_names:
        if config['verbose']:
            print('Generating ' + detector + ' detection training files...')

        output_dir = os.path.join(project, 'detection', detector + '_tfrecords_detection')
        make_clean_dir(output_dir)
        v_info = prep_records_detection(D, detector_list[detector])
        write_to_tfrecord(v_info, output_dir)

        if config['verbose']:
            print('done.')

def prepare_detector_training_data_AL(project, train_length):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    and prior files that are used to train MARS's detectors and pose estimators for black and white mice.
    """
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    detector_list = config['detection']
    detector_names = detector_list.keys()

    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    for detector in detector_names:
        if config['verbose']:
            print('Generating ' + detector + ' detection training files...')

        output_dir = os.path.join(project, 'detection', detector + '_tfrecords_detection')
        make_clean_dir(output_dir)
        v_info = prep_records_detection(D, detector_list[detector])
        write_to_tfrecord_AL(v_info, output_dir, train_length)

        if config['verbose']:
            print('done.')

def prepare_pose_training_data(project):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's pose estimator.
    """
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    pose_list = config['pose']
    pose_names = pose_list.keys()

    # extract info from annotations
    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)
    # random.shuffle(D)

    for pose in pose_names:
        if config['verbose']:
            print('Generating ' + pose + ' pose training files...')

        output_dir = os.path.join(project, 'pose', pose + '_tfrecords_pose')
        make_clean_dir(output_dir)
        if os.path.exists(os.path.join(project, 'annotation_data', 'test_sets')):  # remove old test sets if we had them.
            shutil.rmtree(os.path.join(project, 'annotation_data', 'test_sets'))
        v_info = prep_records_pose(D, pose_list[pose])
        write_to_tfrecord(v_info, output_dir)

        if config['verbose']:
            print('done.')

def prepare_pose_training_data_AL(project, train_length):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's pose estimator.
    """
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    pose_list = config['pose']
    pose_names = pose_list.keys()

    # extract info from annotations
    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)
    # random.shuffle(D)

    for pose in pose_names:
        if config['verbose']:
            print('Generating ' + pose + ' pose training files...')

        output_dir = os.path.join(project, 'pose', pose + '_tfrecords_pose')
        make_clean_dir(output_dir)
        if os.path.exists(os.path.join(project, 'annotation_data', 'test_sets')):  # remove old test sets if we had them.
            shutil.rmtree(os.path.join(project, 'annotation_data', 'test_sets'))
        v_info = prep_records_pose(D, pose_list[pose])
        write_to_tfrecord_AL(v_info, output_dir, train_length)

        if config['verbose']:
            print('done.')

def make_project_priors(project):

    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    detector_list = config['detection']
    detector_names = detector_list.keys()

    for detector in detector_names:
        if config['verbose']:
            print('Generating ' + detector + ' priors...')
        output_dir = os.path.join(project, 'detection', detector + '_tfrecords_detection')
        record_list = glob.glob(os.path.join(output_dir, 'train_dataset-*'))
        priors = generate_priors_from_data(dataset=record_list)

        with open(os.path.join(project, 'detection', 'priors_' + detector + '.pkl'), 'wb') as fp:
            pickle.dump(priors, fp)

        if config['verbose']:
            print('done.')


def annotation_postprocessing(project):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's detectors and pose estimators.
    project : string
        The absolute path to the project directory.
    Example
    --------
    process_annotations('D:\\my_project')
    --------
    """
    # extract info from annotations into an intermediate dictionary file
    make_annot_dict(project)
    # save tfrecords
    prepare_detector_training_data(project)
    prepare_pose_training_data(project)

    # make priors
    make_project_priors(project)
    

def process_active_learning_data(project, frames_included=[], frames_to_add=[],iteration=0):
    """
    Give a .manifest file in /project/annotation_data, create the tfrecord files 
    """
    all_data_path = os.path.join(project, 'annotation_data', 'all_data.manifest')
    init_batch = 100
    with open(all_data_path ) as f:
        frames = f.read().splitlines()
    if not frames_included:
        for _ in range(init_batch):
            frames_included.append(random.randrange(len(frames)))
    new_frames = frames_included + frames_to_add
    new_frames.sort()
    train_manifest = []
    for i in new_frames:
        train_manifest.append(frames[i])
    for i in range(len(frames)):
        if i not in new_frames:
            train_manifest.append(frames[i])
    assert len(train_manifest) == len(frames), 'Train/test data improperly processed.'
    outfile = os.path.join(project, 'annotation_data', str(iteration) + '_data.manifest')
    output = open(outfile, 'w')
    for ex in train_manifest:
        output.write(ex)
        output.write('\n')
    
    # extract info from annotations into an intermediate dictionary file
    make_annot_dict_AL(project, iteration)
    annot_dict = os.path.join(project,'annotation_data','processed_keypoints.json')
    with open(annot_dict) as f:
        keypts = f.read().splitlines()
        for i in range(len(new_frames)):
            assert str(new_frames[i]) in keypts[i]['frame_id'], str(new_frames[i]) + ' is not in processed_keypoints.json (wrong location)'
    
    # save tfrecords
    prepare_detector_training_data_AL(project, len(new_frames))
    prepare_pose_training_data_AL(project, len(new_frames))

    # make priors
    make_project_priors(project)
    print('Data for iteration', iteration, 'processed:', len(train_manifest), 'frames total.')


if __name__ ==  '__main__':
    """
    annotation_postprocessing command line entry point
    Arguments:
        project 	The absolute path to the project directory.
    """

    parser = argparse.ArgumentParser(description='postprocess and package manual pose annotations', prog='annotation_postprocessing')
    parser.add_argument('project', type=str, help="absolute path to project folder.")
    args = parser.parse_args(sys.argv[1:])

    annotation_postprocessing(args.project)
