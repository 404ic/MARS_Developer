import active_learning_pose as AL
import numpy as np

project = '/home/ericma/Documents/active_learning'
frames = AL.AL_pose_choose_worst_images(project, 4)
frames = [int(x) for x in frames]
print(frames)