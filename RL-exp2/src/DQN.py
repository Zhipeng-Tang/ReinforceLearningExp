import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections
import argparse

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('-e', '--env', type=str, default='CartPole-v1', help='which enviroment, CartPole-v1 or MountainCar-v0')
parser.add_argument('-t', '--test', action='store_true', help='test or not test(train)')
parser.add_argument('--double', action='store_true', help='double or not double')
parser.add_argument('--dueling', action='store_true', help='dueling or not dueling')
parser.add_argument('--logdir', type=str, default='./log', help='dir of log')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
parser.add_argument('--epsilon_start', type=float, default=0.5, help='epsilon starting')
args = parser.parse_args()
assert args.env in ['CartPole-v1', 'MountainCar-v0'], 'env: {} does not exist!'.format(args.env)

# hyper-parameters
if args.env == 'CartPole-v1':
    EPISODES = 2000                 # 训练/测试幕数
    BATCH_SIZE = 64
    LR = args.lr
    GAMMA = 0.98
    SAVING_IETRATION = 1000         # 保存Checkpoint的间隔
    MEMORY_CAPACITY = 10000         # Memory的容量
    MIN_CAPACITY = 500              # 开始学习的下限
    Q_NETWORK_ITERATION = 10        # 同步target network的间隔
    EPSILON = 0.01                  # epsilon-greedy
    SEED = 0
    epsilon_start = 0.5
    epsilon_end = 0
    epsilon_decay = EPISODES * 0.8
elif args.env == 'MountainCar-v0':
    EPISODES = 2000                 # 训练/测试幕数
    BATCH_SIZE = 64
    LR = args.lr
    GAMMA = 0.98
    SAVING_IETRATION = 1000         # 保存Checkpoint的间隔
    MEMORY_CAPACITY = 10000         # Memory的容量
    MIN_CAPACITY = 500              # 开始学习的下限
    Q_NETWORK_ITERATION = 10        # 同步target network的间隔
    EPSILON = 0.01                  # epsilon-greedy
    SEED = 0
    epsilon_start = args.epsilon_start
    epsilon_end = 0
    epsilon_decay = EPISODES * 0.8

TEST = args.test
if not args.double:
    SAVE_PATH_PREFIX = '{}/{}/dqn/'.format(args.logdir, args.env) if not args.dueling else '{}/{}/dueling_dqn/'.format(args.logdir, args.env)
    MODEL_PATH = '{}/{}/dqn/ckpt/final.pth'.format(args.logdir, args.env) if not args.dueling else '{}/{}/dueling_dqn/ckpt/final.pth'.format(args.logdir, args.env)
else:
    SAVE_PATH_PREFIX = '{}/{}/double_dqn/'.format(args.logdir, args.env) if not args.dueling else '{}/{}/double_dueling_dqn/'.format(args.logdir, args.env)
    MODEL_PATH = '{}/{}/double_dqn/ckpt/final.pth'.format(args.logdir, args.env) if not args.dueling else '{}/{}/double_dueling_dqn/ckpt/final.pth'.format(args.logdir, args.env)


env = gym.make(args.env, render_mode="human" if TEST else None)
# env = gym.make('CartPole-v1', render_mode="human" if TEST else None)
# env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)


random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

NUM_ACTIONS = env.action_space.n  # 2
NUM_STATES = env.observation_space.shape[0]  # 4
ENV_A_SHAPE = 0 if np.issubdtype(type(env.action_space.sample()), int) else env.action_space.sample().shape  # 0, to confirm the shape

class DQNModel(nn.Module):
    def __init__(self, num_inputs=4):
        super(DQNModel, self).__init__()
        self.linear = nn.Linear(NUM_STATES, 512)
        self.linear2 = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class DuelingDQNModel(nn.Module):
    def __init__(self):
        super(DuelingDQNModel, self).__init__()
        self.linear = nn.Linear(NUM_STATES, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, NUM_ACTIONS)
    
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        V = self.V(x)
        A = self.A(x)
        return V, A


class Data:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data, index):
        # TODO
        if len(self.buffer) < MEMORY_CAPACITY:
            self.buffer.append(data)
        else:
            self.buffer[index] = data
    
    def get(self, batch_size):
        # TODO
        batch = random.sample(self.buffer, batch_size)
        return batch


class DQN():
    """docstring for DQN"""
    def __init__(self, is_double=False):
        '''
        is_double: double DQN or not
        '''
        super(DQN, self).__init__()
        self.generate_net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.is_double = is_double
    
    def generate_net(self):
        self.eval_net, self.target_net = DQNModel().to(device), DQNModel().to(device)

    def calc_eval_action_values(self, state):
        return self.eval_net.forward(state)
    
    def calc_target_action_values(self, state):
        return self.target_net.forward(state)

    def choose_action(self, state, EPSILON = 1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            with torch.no_grad():
                action_value = self.calc_eval_action_values(state)
                action = torch.argmax(action_value).item()
                action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        else: 
            # random policy
            action = np.random.randint(0,NUM_ACTIONS)  # int random number
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory.set(data, self.memory_counter % MEMORY_CAPACITY)
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        if self.learn_step_counter % 1000 == 0:
            print('training step : {}'.format(self.learn_step_counter))
        self.learn_step_counter += 1

        # TODO
        batch = self.memory.get(BATCH_SIZE)

        curr_states = torch.tensor(np.array([data.state for data in batch]), dtype=torch.float).to(device)
        curr_actions = torch.tensor(np.array([data.action for data in batch]), dtype=torch.int64).to(device)
        reward = torch.tensor(np.array([data.reward for data in batch]), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array([data.next_state for data in batch]), dtype=torch.float).to(device)
        dones = torch.tensor(np.array([data.done for data in batch]), dtype=torch.float).to(device)
        curr_action_values: torch.Tensor = self.calc_eval_action_values(curr_states)
        next_action_values: torch.Tensor = self.calc_target_action_values(next_states)

        choices = curr_action_values.gather(1, curr_actions.view(BATCH_SIZE,1)).view(BATCH_SIZE)
        if not self.is_double:
            targets = reward + torch.max(next_action_values,dim=1).values * GAMMA * (1- dones)
        else:
            next_action_values_using_evalnet = self.calc_eval_action_values(next_states)
            targets = reward + next_action_values.gather(1, torch.argmax(next_action_values_using_evalnet,dim=1).view(BATCH_SIZE, 1)).view(BATCH_SIZE) * GAMMA * (1- dones)

        loss = self.loss_func(choices, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))

class DuelingDQN(DQN):
    """docstring for DuelingDQN"""
    def __init__(self):
        super(DuelingDQN, self).__init__()

    def generate_net(self):
        self.eval_net, self.target_net = DuelingDQNModel().to(device), DuelingDQNModel().to(device)

    def calc_eval_action_values(self, state):
        V, A = self.eval_net.forward(state)
        action_values = V + A - A.mean(dim=-1, keepdim=True)
        return action_values
    
    def calc_target_action_values(self, state):
        V, A = self.target_net.forward(state)
        action_values = V + A - A.mean(dim=-1, keepdim=True)
        return action_values

def main():
    if args.dueling:
        dqn = DuelingDQN(args.double)
    else:
        dqn = DQN(args.double)
    
    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')

    if TEST:
        dqn.load_net(MODEL_PATH)
    for i in range(EPISODES):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED)

        ep_reward = 0
        EPSILON = epsilon_end + (epsilon_start - epsilon_end) * (1 - (i + 1) / epsilon_decay)
        while True:
            action = dqn.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)  # choose best action
            next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            # if TEST:
            #     env.render()
            if dqn.memory_counter >= MIN_CAPACITY and not TEST:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                if TEST:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break
            state = next_state
        writer.add_scalar('reward', ep_reward, global_step=i)
    if not TEST:
        dqn.save_train_model('final')


if __name__ == '__main__':
    main()