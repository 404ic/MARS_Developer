import multibox_detection.train_detect as train_detector
import multibox_detection.evaluate_detection as evaluate_detector
from create_new_project import *

import os

def create_AL_project(project_path):
    # makes project at project_path/active_learning 
    project_dir = os.path.join(project_path, 'active_learning')
    os.mkdir(project_dir)
    print('Directory at ' + project_dir + ' created')

