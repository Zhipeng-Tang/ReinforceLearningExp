import os
import argparse

parser = argparse.ArgumentParser(description='running script of DQN')
parser.add_argument('-e', '--env', type=str, default='CartPole-v1', help='which enviroment, CartPole-v1 or MountainCar-v0')
parser.add_argument('-d', '--device', type=int, default=0, help='which GPU to use')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
parser.add_argument('--epsilon_start', type=float, default=0.5, help='epsilon starting')
args = parser.parse_args()
assert args.env in ['CartPole-v1', 'MountainCar-v0']
assert 0 <= args.device <= 7

cmds = ['CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --lr {} --epsilon_start {}'.format(args.device, args.env, args.lr, args.epsilon_start),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --double --lr {} --epsilon_start {}'.format(args.device, args.env, args.lr, args.epsilon_start),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --dueling --lr {} --epsilon_start {}'.format(args.device, args.env, args.lr, args.epsilon_start),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} --double --dueling --lr {} --epsilon_start {}'.format(args.device, args.env, args.lr, args.epsilon_start),]

for cmd in cmds:
    os.system(cmd)