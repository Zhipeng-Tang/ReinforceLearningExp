import matplotlib.pyplot as plt
import argparse
import os
from tensorboard.backend.event_processing import event_accumulator

def smooth(rewards: list, weight=0.99) -> list:
    assert len(rewards) > 0, 'rewards must not be empty!'
    smooth_rewards = []
    history = rewards[0]
    for reward in rewards:
        history = history * weight + reward * (1 - weight)
        smooth_rewards.append(history)
    return smooth_rewards

parser = argparse.ArgumentParser(description='plot reward curve')
parser.add_argument('-d', '--dir', type=str, default='./log', help='dir to plot')
parser.add_argument('-o', '--output_name', type=str, default='output.png', help='name of output file')
parser.add_argument('-s', '--smooth', type=float, default=0.99, help='smoothing weight')
args = parser.parse_args()

assert os.path.exists(args.dir), 'dir {} not exist!'.format(args.dir)
assert args.output_name[-4:] == '.png', 'output file must be png file!'

modes = os.listdir(args.dir)

plt.figure(figsize=(10, 6))
for mode in modes:
    path = os.path.join(args.dir, mode)
    filename = list(filter(lambda file: 'events' in file, os.listdir(path)))[-1]
    filepath = os.path.join(path, filename)
    ea = event_accumulator.EventAccumulator(filepath)
    ea.Reload()

    steps = [item.step for item in ea.scalars.Items('reward')]
    rewards = [item.value for item in ea.scalars.Items('reward')]

    smooth_rewards = smooth(rewards, args.smooth)

    plt.plot(steps, smooth_rewards, label=mode)
plt.xlabel('step')
plt.ylabel('reward')
plt.legend()
plt.savefig(args.output_name, dpi=600)