import os

if __name__ == '__main__':
    env_names = ['halfcheetah-medium-expert-v0', 'halfcheetah-medium-v0',
                 'hopper-random-v0']

    for env in env_names:
        for use in ['true', 'false']:
            for reg in ['kl', 'mmd']:
                for seed in [200, 201, 202]:
                    for file in ['behavior.ckpt.index', 'behavior.ckpt.data-00000-of-00001']:
                        if use != 'true' or reg != 'kl':
                            src_folder = f'env\={env}_reg\=kl_use\=true/env\={env}_reg\=kl_use\=true_s{seed}/'
                            des_folder = f'env\={env}_reg\={reg}_use\={use}/env\={env}_reg\={reg}_use\={use}_s{seed}/'
                            command = 'cp ' + src_folder + file + ' ' + des_folder
                            # print(command)
                            # os.system(command)

    for env in env_names:
        for reg in ['kl', 'mmd']:
            for seed in [200, 201, 202]:
                for file in [f'policy_behavior_None_{reg}.ckpt.data-00000-of-00001',
                             f'policy_behavior_None_{reg}.ckpt.index',
                             f'policy_behavior_log_beta_None_{reg}.ckpt.data-00000-of-00001',
                             f'policy_behavior_log_beta_None_{reg}.ckpt.index']:
                    src_folder = f'env\={env}_reg\={reg}_use\=true/env\={env}_reg\={reg}_use\=true_s{seed}/'
                    des_folder = f'env\={env}_reg\={reg}_use\=false/env\={env}_reg\={reg}_use\=false_s{seed}/'
                    command = 'cp ' + des_folder + file + ' ' + src_folder
                    # print(command)
                    os.system(command)
