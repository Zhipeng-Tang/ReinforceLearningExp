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

parser= argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dqn')
parser.add_argument('--double_dqn', action='store_true')
parser.add_argument('--task', type=str, default='CartPole-v1')
args = parser.parse_args()

# hyper-parameters
EPISODES = 2000                 # 训练/测试幕数
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_IETRATION = 1000         # 保存Checkpoint的间隔
MEMORY_CAPACITY = 10000         # Memory的容量
MIN_CAPACITY = 500              # 开始学习的下限
Q_NETWORK_ITERATION = 10        # 同步target network的间隔
EPSILON = 0.01                  # epsilon-greedy
SEED = 0
MODEL_PATH = ''
SAVE_PATH_PREFIX = './log/dqn/'
TEST = False


env = gym.make(args.task, render_mode="human" if TEST else None)
# env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)


random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

NUM_ACTIONS = env.action_space.n  # 2
NUM_STATES = env.observation_space.shape[0]  # 4
ENV_A_SHAPE = 0 if np.issubdtype(type(env.action_space.sample()), np.integer) else env.action_space.sample().shape  # 0, to confirm the shape

class Model(nn.Module):
    def __init__(self, num_inputs=4):
        super(Model, self).__init__()
        self.linear = nn.Linear(NUM_STATES, 512)
        self.linear2 = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


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
        self.capacity = capacity

    def set(self, data,index):
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[index] = data
    
    def get(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
        


class DQN():
    """docstring for DQN"""
    def __init__(self,is_double=False):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Model().to(device), Model().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.generate_net()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.is_double=is_double

    def generate_net(self):
        self.eval_net = Model().to(device)
        self.target_net = Model().to(device)

    def calc_eval_action_values(self, state):
        return self.eval_net.forward(state)
    
    def calc_target_action_values(self, state):
        return self.target_net.forward(state)

    def choose_action(self, state, EPSILON = 1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            action_value = self.calc_eval_action_values(state)
            action = torch.argmax(action_value).item()
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        else: 
            # random policy
            action = np.random.randint(0,NUM_ACTIONS)  # int random number
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory.set(data,self.memory_counter % self.memory.capacity)
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        batch = self.memory.get(BATCH_SIZE)

        curr_states = torch.tensor(np.array([data.state for data in batch]), dtype=torch.float).to(device)
        curr_actions = torch.tensor(np.array([data.action for data in batch]), dtype=torch.int64).to(device)
        rewards = torch.tensor(np.array([data.reward for data in batch]), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array([data.next_state for data in batch]), dtype=torch.float).to(device)
        dones = torch.tensor(np.array([data.done for data in batch]), dtype=torch.float).to(device)
        curr_action_values: torch.Tensor = self.calc_eval_action_values(curr_states)
        next_action_values: torch.Tensor = self.calc_target_action_values(next_states)

        choices = curr_action_values.gather(1, curr_actions.view(BATCH_SIZE,1)).view(BATCH_SIZE)
        if not self.is_double:
            targets = rewards + torch.max(next_action_values,dim=1).values * GAMMA * (1 - dones)
        else:
            next_action_values_using_evalnet = self.calc_eval_action_values(next_states)
            targets = rewards + next_action_values.gather(1, torch.argmax(next_action_values_using_evalnet,dim=1).view(BATCH_SIZE, 1)).view(BATCH_SIZE) * GAMMA * (1 - dones)

        loss = self.loss_func(choices, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))

class DuelingModel(nn.Module):
    '''
    Model of Dueling DQN.
    '''
    def __init__(self):
        super(DuelingModel, self).__init__()
        self.linear = nn.Linear(NUM_STATES, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        V = self.V(x)
        A = self.A(x)
        return V, A

class DuelingDQN(DQN):
    def __init__(self, is_double=False,):
        super(DuelingDQN, self).__init__(is_double)

    def generate_net(self):
        self.eval_net = DuelingModel().to(device)
        self.target_net = DuelingModel().to(device)

    def calc_eval_action_values(self, state):
        V, A = self.eval_net.forward(state)
        action_values = V + A - A.mean(dim=-1, keepdim=True)
        return action_values
    
    def calc_target_action_values(self, state):
        V, A = self.target_net.forward(state)
        action_values = V + A - A.mean(dim=-1, keepdim=True)
        return action_values


def main():
    if args.mode == 'dqn':
        dqn = DQN(args.double_dqn)
    else:
        dqn = DuelingDQN(args.double_dqn)
    
    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')

    if TEST:
        dqn.load_net(MODEL_PATH)
    for i in range(EPISODES):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED)

        ep_reward = 0
        while True:
            action = dqn.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)  # choose best action
            next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            if TEST:
                env.render()
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


if __name__ == '__main__':
    main()