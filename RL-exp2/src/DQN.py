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
parser.add_argument('-m', '--mode', type=str, default='dqn', help='dqn, dueling or prioritized_relay')
parser.add_argument('-t', '--test', action='store_true', help='test or not test(train)')
parser.add_argument('--double', action='store_true', help='double or not double')
parser.add_argument('--noisy', action='store_true', help='use noisy-net or not')
parser.add_argument('--logdir', type=str, default='./log', help='dir of log')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate, default is 0.00025')
parser.add_argument('--epsilon_start', type=float, default=0.5, help='epsilon starting, default is 0.5')
parser.add_argument('--omega', type=float, default=0.5, help='a hyper-parameter of prioritized relay dqn, default is 0.5')
parser.add_argument('--dynamic_epsilon', action='store_true', help='use dynamic epsilon or not')
parser.add_argument('--multi_step', type=int, default=1, help='how many steps for multi-step learning, default is 1')
args = parser.parse_args()
assert args.env in ['CartPole-v1', 'MountainCar-v0'], 'env: {} does not exist!'.format(args.env)
assert args.mode in ['dqn', 'dueling', 'prioritized_relay'], 'mode: {} does not exist!'.format(args.mode)
assert args.multi_step >= 1, 'multi_step must be greater than or equal to 1'

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

dirname = 'double_{}'.format(args.mode) if args.double else args.mode
dirname = 'noisy_{}'.format(dirname) if args.noisy else dirname
dirname = 'multi_step_{}_{}'.format(args.multi_step, dirname) if args.multi_step > 1 else dirname
SAVE_PATH_PREFIX = '{}/{}/{}/'.format(args.logdir, args.env, dirname)
MODEL_PATH = '{}/{}/{}/ckpt/final.pth'.format(args.logdir, args.env, dirname)


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

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = (3 / self.in_features) ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    def forward(self, x):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return F.linear(x, weight, bias)

class DQNModel(nn.Module):
    def __init__(self, in_features, out_features, is_noisy=False):
        super(DQNModel, self).__init__()
        if not is_noisy:
            self.linear1 = nn.Linear(in_features, 512)
            self.linear2 = nn.Linear(512, out_features)
        else:
            self.linear1 = NoisyLinear(in_features, 512)
            self.linear2 = NoisyLinear(512, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class DuelingDQNModel(nn.Module):
    def __init__(self, in_features, out_features, is_noisy):
        super(DuelingDQNModel, self).__init__()
        if not is_noisy:
            self.linear = nn.Linear(in_features, 512)
            self.V = nn.Linear(512, 1)
            self.A = nn.Linear(512, out_features)
        else:
            self.linear = NoisyLinear(in_features, 512)
            self.V = NoisyLinear(512, 1)
            self.A = NoisyLinear(512, out_features)
    
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
        self.capacity = capacity

    def set(self, data, index):
        # TODO
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[index] = data
    
    def get(self, batch_size, probability=None):
        '''
        Choose batch_size elements from buffer according to probability. \n
        If probability is not specified, the choices are made with equal probability.
        '''
        # TODO
        if probability is not None:
            batch = random.choices(self.buffer, probability, k=batch_size)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch


class DQN():
    def __init__(self, 
                 num_states, 
                 num_actions, 
                 env_a_shape, 
                 multi_step=1, 
                 is_double=False, 
                 is_noisy=False,
                 batch_size=64, 
                 memory_capacity=10000, 
                 gamma=0.98, 
                 q_netwotk_iteration=10, 
                 saving_iteration=1000):
        '''
        num_states: number of states
        num_actions: number of actions
        env_shape: shape of action space
        is_double: double DQN or not
        is_noisy: use noisy-net or not
        batch_size: batch_size
        memory_capacity: capacity of memory
        gamma: gamma
        q_network_iteration: for each q_network_iteration iterations, target_net <- eval_net
        saving_iteration: for each saving_iteration iterations, save model
        '''
        super(DQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=memory_capacity)
        self.loss_func = nn.MSELoss()
        self.is_double = is_double
        self.is_noisy = is_noisy
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_network_iteration=q_netwotk_iteration
        self.saving_iteration = saving_iteration
        self.env_a_shape = env_a_shape
        self.multi_step = multi_step
        self.generate_net()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
    
    def generate_net(self):
        self.eval_net = DQNModel(self.num_states, self.num_actions, self.is_noisy).to(device)
        self.target_net = DQNModel(self.num_states, self.num_actions, self.is_noisy).to(device)

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
                action = action if self.env_a_shape ==0 else action.reshape(self.env_a_shape)
        else: 
            # random policy
            action = np.random.randint(0,self.num_actions)  # int random number
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action

    def store_transition(self, data):
        length = len(data)
        assert self.multi_step == length, 'multi_step is {}, but the length of data is {}, not match!'.format(self.multi_step, length)
        reward = np.sum(np.array([d.reward for d in data]) * np.array([self.gamma ** i for i in range(length)]))
        self.memory.set(Data(data[0].state, data[0].action, reward, data[length-1].next_state, data[length-1].done), self.memory_counter % self.memory.capacity)
        self.memory_counter += 1

    def get_batch(self, batch_size):
        return self.memory.get(batch_size)
    
    def learn(self):
        # update the parameters
        if self.learn_step_counter % self.q_network_iteration ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % self.saving_iteration == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # TODO
        batch = self.get_batch(self.batch_size)

        curr_states = torch.tensor(np.array([data.state for data in batch]), dtype=torch.float).to(device)
        curr_actions = torch.tensor(np.array([data.action for data in batch]), dtype=torch.int64).to(device)
        rewards = torch.tensor(np.array([data.reward for data in batch]), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array([data.next_state for data in batch]), dtype=torch.float).to(device)
        dones = torch.tensor(np.array([data.done for data in batch]), dtype=torch.float).to(device)
        curr_action_values: torch.Tensor = self.calc_eval_action_values(curr_states)
        next_action_values: torch.Tensor = self.calc_target_action_values(next_states)

        choices = curr_action_values.gather(1, curr_actions.view(self.batch_size,1)).view(self.batch_size)
        if not self.is_double:
            targets = rewards + torch.max(next_action_values,dim=1).values * (self.gamma ** self.multi_step) * (1 - dones)
        else:
            next_action_values_using_evalnet = self.calc_eval_action_values(next_states)
            targets = rewards + next_action_values.gather(1, torch.argmax(next_action_values_using_evalnet,dim=1).view(self.batch_size, 1)).view(self.batch_size) * (self.gamma ** self.multi_step) * (1 - dones)

        loss = self.loss_func(choices, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}/ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))

class DuelingDQN(DQN):
    def __init__(self, 
                 num_states, 
                 num_actions,
                 env_a_shape, 
                 multi_step=1, 
                 is_double=False, 
                 is_noisy=False,
                 batch_size=64, 
                 memory_capacity=10000, 
                 gamma=0.98, 
                 q_netwotk_iteration=10, 
                 saving_iteration=1000):
        super(DuelingDQN, self).__init__(num_states, 
                                         num_actions, 
                                         env_a_shape, 
                                         multi_step, 
                                         is_double, 
                                         is_noisy, 
                                         batch_size, 
                                         memory_capacity, 
                                         gamma, 
                                         q_netwotk_iteration, 
                                         saving_iteration)

    def generate_net(self):
        self.eval_net = DuelingDQNModel(self.num_states, self.num_actions, self.is_noisy).to(device)
        self.target_net = DuelingDQNModel(self.num_states, self.num_actions, self.is_noisy).to(device)

    def calc_eval_action_values(self, state):
        V, A = self.eval_net.forward(state)
        action_values = V + A - A.mean(dim=-1, keepdim=True)
        return action_values
    
    def calc_target_action_values(self, state):
        V, A = self.target_net.forward(state)
        action_values = V + A - A.mean(dim=-1, keepdim=True)
        return action_values

class PrioritizedRelayDQN(DQN):
    def __init__(self, 
                 num_states, 
                 num_actions,
                 env_a_shape,  
                 omega, 
                 multi_step=1, 
                 is_double=False, 
                 is_noisy=False,
                 batch_size=64, 
                 memory_capacity=10000, 
                 gamma=0.98, 
                 q_netwotk_iteration=10, 
                 saving_iteration=1000):
        '''
        omega: a hyper-parameter that determines the shape of distribution.
        '''
        super(PrioritizedRelayDQN, self).__init__(num_states, 
                                                  num_actions, 
                                                  env_a_shape, 
                                                  multi_step, 
                                                  is_double, 
                                                  is_noisy, 
                                                  batch_size, 
                                                  memory_capacity, 
                                                  gamma, 
                                                  q_netwotk_iteration, 
                                                  saving_iteration)
        self.omega = omega
    
    def get_batch(self, batch_size):
        curr_states = torch.tensor(np.array([data.state for data in self.memory.buffer]), dtype=torch.float).to(device)
        curr_actions = torch.tensor(np.array([data.action for data in self.memory.buffer]), dtype=torch.int64).to(device)
        reward = torch.tensor(np.array([data.reward for data in self.memory.buffer]), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array([data.next_state for data in self.memory.buffer]), dtype=torch.float).to(device)
        dones = torch.tensor(np.array([data.done for data in self.memory.buffer]), dtype=torch.float).to(device)
        curr_action_values: torch.Tensor = self.calc_eval_action_values(curr_states)
        next_action_values: torch.Tensor = self.calc_target_action_values(next_states)

        choices = curr_action_values.gather(1, curr_actions.view(len(self.memory.buffer),1)).view(len(self.memory.buffer))
        targets = reward + torch.max(next_action_values,dim=1).values * self.gamma * (1- dones)

        probability = torch.float_power(torch.abs(targets - choices), self.omega)
        probability = (probability / torch.sum(probability)).tolist()

        return self.memory.get(batch_size, probability)

def main():
    if args.mode == 'dqn':
        dqn = DQN(NUM_STATES, NUM_ACTIONS, ENV_A_SHAPE, args.multi_step, args.double, args.noisy, BATCH_SIZE, MEMORY_CAPACITY, GAMMA, Q_NETWORK_ITERATION, SAVING_IETRATION)
    elif args.mode == 'dueling':
        dqn = DuelingDQN(NUM_STATES, NUM_ACTIONS, ENV_A_SHAPE, args.multi_step, args.double, args.noisy, BATCH_SIZE, MEMORY_CAPACITY, GAMMA, Q_NETWORK_ITERATION, SAVING_IETRATION)
    elif args.mode == 'prioritized_relay':
        dqn = PrioritizedRelayDQN(NUM_STATES, NUM_ACTIONS, ENV_A_SHAPE, args.omega, args.multi_step, args.double, args.noisy, BATCH_SIZE, MEMORY_CAPACITY, GAMMA, Q_NETWORK_ITERATION, SAVING_IETRATION)
    
    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')

    if TEST:
        dqn.load_net(MODEL_PATH)
    for i in range(EPISODES):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED)

        ep_reward = 0
        if args.dynamic_epsilon:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - (i + 1) / epsilon_decay)
        else:
            epsilon = EPSILON
        
        counter = 0
        data_list = [None for _ in range(args.multi_step)]
        done = False
        for _ in range(args.multi_step-1):
            action = dqn.choose_action(state=state, EPSILON=epsilon if not TEST else 0)  # choose best action
            next_state, reward, terminated, truncated, info = env.step(action)  # observe next state and reward
            done = terminated or truncated
            data_list[counter % args.multi_step] = Data(state, action, reward, next_state, done)
            ep_reward += reward
            counter += 1
            state = next_state
            if done:
                print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break
        if not done:
            while True:
                action = dqn.choose_action(state=state, EPSILON=epsilon if not TEST else 0)  # choose best action
                next_state, reward, terminated, truncated, info = env.step(action)  # observe next state and reward
                done = terminated or truncated
                data_list[counter % args.multi_step] = Data(state, action, reward, next_state, done)
                counter += 1
                dqn.store_transition([data_list[i % args.multi_step] for i in range(counter - args.multi_step, counter)])
                ep_reward += reward
                state = next_state
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
        writer.add_scalar('reward', ep_reward, global_step=i)
    if not TEST:
        dqn.save_train_model('final')


if __name__ == '__main__':
    main()