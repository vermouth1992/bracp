"""
Run ablation study.
1. KLD vs MMD
2. GP vs. no GP
We run for 6 tasks, each task runs for 3 seeds. In total, we run 72 configs.
"""

import rlutils.infra as rl_infra


def thunk(**args):
    import sys
    import d4rl
    import os.path as osp
    sys.path.append(osp.abspath(osp.dirname(__file__)))
    from bracp import BRACPRunner
    # add default arguments
    reg_type = args['reg_type']
    env_name = args['env_name']

    [].append(d4rl)

    default_args = {
        'hopper-medium-expert-v0': {
            'generalization_threshold': 0.2 if reg_type == 'kl' else 0.02,
        },
        'walker2d-medium-expert-v0': {
            'generalization_threshold': 0.2 if reg_type == 'kl' else 0.02,
        },
        'halfcheetah-medium-expert-v0': {
            'generalization_threshold': 0.2 if reg_type == 'kl' else 0.02,
            'max_ood_grad_norm': 0.1,
        },
        'hopper-medium-v0': {
            'generalization_threshold': 0.2 if reg_type == 'kl' else 0.02,
        },
        'walker2d-medium-v0': {
            'generalization_threshold': 1.0 if reg_type == 'kl' else 0.1,
        },
        'halfcheetah-medium-v0': {
            'generalization_threshold': 7.0 if reg_type == 'kl' else 0.7,
            'max_ood_grad_norm': 0.1,
        },
        'hopper-medium-replay-v0': {
            'generalization_threshold': 3.0 if reg_type == 'kl' else 0.3,
        },
        'walker2d-medium-replay-v0': {
            'generalization_threshold': 3.0 if reg_type == 'kl' else 0.3,
        },
        'halfcheetah-medium-replay-v0': {
            'generalization_threshold': 3.0 if reg_type == 'kl' else 0.3,
            'max_ood_grad_norm': 0.1,
        },
        'hopper-random-v0': {
            'generalization_threshold': 3.0 if reg_type == 'kl' else 0.3,
        },
        'walker2d-random-v0': {
            'generalization_threshold': 3.0 if reg_type == 'kl' else 0.3,
        },
        'halfcheetah-random-v0': {
            'generalization_threshold': 3.0 if reg_type == 'kl' else 0.3,
            'max_ood_grad_norm': 0.1,
        },
    }
    override_args = default_args.get(env_name, dict())
    args.update(override_args)
    BRACPRunner.main(**args)


if __name__ == '__main__':
    # env_names = ['hopper-medium-replay-v0', 'hopper-random-v0', 'walker2d-medium-v0']
    env_names = ['walker2d-medium-expert-v0', 'halfcheetah-medium-expert-v0', 'halfcheetah-medium-v0']
    experiments = rl_infra.runner.ExperimentGrid()
    experiments.add(key='env_name', vals=env_names)
    experiments.add(key='reg_type', vals=['kl', 'mmd'], in_name=True)
    experiments.add(key='use_gp', vals=[True, False], in_name=True)
    experiments.add(key='seed', vals=[200, 201, 202])
    experiments.add(key='epochs', vals=200)  # run half of the normal length to save time
    experiments.run(thunk=thunk, data_dir='data/ablation_results')
