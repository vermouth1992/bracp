# BRAC+: Improved Behavior Regularized Offline Reinforcement Learning
This repository is the official implementation of [BRAC+: Improved Behavior Regularized Actor Critic for Offline Reinforcement Learning](https://arxiv.org/abs/2110.00894).
## Requirements
We high recommend that you create a new Python environment to test our code
#### Conda Environment
```shell
conda create -n bracp python=3.8
```

To install requirements:
#### Python package
```shell
pip install -r requirements.txt
```

#### D4RL library
```shell
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

#### rlutils library
```shell
pip install rlutils-python==0.0.3
```

## Training
```shell
python d4rl_bracp.py train --env_name halfcheetah-medium-v0 --seed 110
```
The script will first pretrain the behavior policy and the initial policy that minimize the KL divergence.

## Logging
The logs will be placed at data/d4rl_results/
