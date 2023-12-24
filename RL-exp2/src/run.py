import os
envs = ['CartPole-v1', 'MountainCar-v0']
modes = ['normal', 'double']

cmds = ['CUDA_VISIBLE_DEVICES=2 python DQN.py -m normal -e CartPole-v1',
        'CUDA_VISIBLE_DEVICES=2 python DQN.py -m double -e CartPole-v1',
        'CUDA_VISIBLE_DEVICES=2 python DQN.py -m normal -e CartPole-v1 -d',
        'CUDA_VISIBLE_DEVICES=2 python DQN.py -m double -e CartPole-v1 -d',]

for cmd in cmds:
    os.system(cmd)