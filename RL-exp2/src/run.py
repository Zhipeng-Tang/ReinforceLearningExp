import os
import argparse

parser = argparse.ArgumentParser(description='running script of DQN')
parser.add_argument('-e', '--env', type=str, default='CartPole-v1', help='which enviroment, CartPole-v1 or MountainCar-v0')
parser.add_argument('-d', '--device', type=int, default=0, help='which GPU to use')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
parser.add_argument('--epsilon_start', type=float, default=0.5, help='epsilon starting')
parser.add_argument('--dynamic_epsilon', action='store_true', help='use dynamic epsilon or not')
args = parser.parse_args()
assert args.env in ['CartPole-v1', 'MountainCar-v0']
assert 0 <= args.device <= 7

# cmds = ['CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dqn --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dqn --double --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dueling --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dueling --double --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m prioritized_relay --double --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dqn --noisy --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dqn --multi_step 3 --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dqn --multi_step 5 --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
#         'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m categorical --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else '')]

cmds = ['CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dqn --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dqn --double --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dueling --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),
        'CUDA_VISIBLE_DEVICES={} python DQN.py -e {} -m dueling --double --lr {} --epsilon_start {} {}'.format(args.device, args.env, args.lr, args.epsilon_start, '--dynamic_epsilon' if args.dynamic_epsilon else ''),]

for cmd in cmds:
    os.system(cmd)