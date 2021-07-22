from hourglass_pose.evaluate_pose import *
from hourglass_pose.config import parse_config_file

def evaluate_all_checkpoints(project):
    log = os.path.join(project, 'pose', 'top_log')
    ckpt_nums = glob.glob(os.path.join(log, 'model.ckpt-*'))
    ckpt_nums = [num for num in ckpt_nums if 'index' not in num and 'data' not in num]
    ckpt_nums = [int(num.split('-')[1].split('.')[0]) for num in ckpt_nums]
    ckpt_nums.sort()
    print(ckpt_nums[34:-1])
    ckpt_nums = ckpt_nums[34:-1]
    for ckpt_num in ckpt_nums:
        print('============================================================')
        print('Processing model', ckpt_num)
        run_test(project, ckpt_num=ckpt_num)
        plot_model_PCK(project, ckpt_num=ckpt_num)
        plot_score_vs_accuracy(project=project, ckpt_num=ckpt_num)
        print('============================================================')
    return

project = '/home/ericma/Documents/reproduce_pose'
evaluate_all_checkpoints(project)