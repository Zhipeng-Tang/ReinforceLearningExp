import os
import argparse

parser = argparse.ArgumentParser(description='running script of DQN')
parser.add_argument('-e', '--env', type=str, default='CartPole-v1', help='which enviroment, CartPole-v1 or MountainCar-v0')
parser.add_argument('-d', '--device', type=int, default=0, help='which GPU to use')
args = parser.parse_args()
assert args.env in ['CartPole-v1', 'MountainCar-v0']
assert 0 <= args.device <= 7

cmds = ['CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --epsilon_start 0.95'.format(args.device, args.env),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --double --epsilon_start 0.95'.format(args.device, args.env),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --dueling --epsilon_start 0.95'.format(args.device, args.env),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --double --dueling --epsilon_start 0.95'.format(args.device, args.env),]

for cmd in cmds:
    os.system(cmd)