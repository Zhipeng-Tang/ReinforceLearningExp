<h1><center>RL-exp2</center></h1>
<center>BY  唐志鹏  SA23011068</center>

## 1 实验内容
- 实现 DQN, Double DQN, Dueling DQN, Dueling Double DQN 算法
- 在两个仿真环境下进行训练，并比较不同算法的性能
- 本文还实现了 multi-step learning, noisy-net, prioritized relay, categorical dqn

## 2 原理与实现
源码参见 [DQN](https://github.com/Zhipeng-Tang/ReinforceLearningExp/blob/master/RL-exp2/src/DQN.py)
### 2.1 DQN
#### 2.1.1 算法原理
- step1: 在环境中采取行动 $a_i$，然后将 $(s_i, a_i, s_{i+1}, r_i)$ 插入 Experience Relay
- step2: 如果 Experience Relay 中有足够的数据，则随机采样 $(s_i, a_i, s_{i+1}, r_i)$ 进行训练
- step3: ${\rm target}(s_i) = r_i + \gamma {\rm max}_{a'}Q_{\phi^-}(s_{i+1}, a')$
- step4: $\phi \leftarrow \phi + \alpha ({\rm target(s_i) - Q_{\phi}(s_i,a_i)}) \frac{{\rm d} Q_{\phi}}{{\rm d} \phi}$
- step5: 每训练 N 步, 令 $\phi^- = \phi$
- 重复上述步骤

#### 2.1.2 实现
- 首先是 Expericence Relay 的实现
  ```python
  class Memory:
    def __init__(self, capacity):
        '''
        capacity: max capacity of memory
        '''
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def set(self, data, index):
        # 如果 memory 还未满，直接将 data 放入 memory 中
        # 否则，将 data 放入 index 处，替换最旧的数据
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[index] = data
    
    def get(self, batch_size):
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        return batch
  ```
- 然后是 DQN 的实现
  ```python
  def learn(self):
    # update the parameters
    if self.learn_step_counter % self.q_network_iteration ==0:
        self.target_net.load_state_dict(self.eval_net.state_dict())
    if self.learn_step_counter % self.saving_iteration == 0:
        self.save_train_model(self.learn_step_counter)

    self.learn_step_counter += 1

    # 随机采样数据
    batch = self.get_batch(self.batch_size)

    # 将数据转换为 torch.tensor
    curr_states = torch.tensor(np.array([data.state for data in batch]), dtype=torch.float).to(device)
    curr_actions = torch.tensor(np.array([data.action for data in batch]), dtype=torch.int64).to(device)
    rewards = torch.tensor(np.array([data.reward for data in batch]), dtype=torch.float).to(device)
    next_states = torch.tensor(np.array([data.next_state for data in batch]), dtype=torch.float).to(device)
    dones = torch.tensor(np.array([data.done for data in batch]), dtype=torch.float).to(device)

    curr_action_values: torch.Tensor = self.calc_eval_action_values(curr_states)
    next_action_values: torch.Tensor = self.calc_target_action_values(next_states)

    # 计算网络估计的 reward 的期望
    choices = curr_action_values.gather(1, curr_actions.view(self.batch_size,1)).view(self.batch_size)
    
    # 计算真值
    targets = rewards + torch.max(next_action_values,dim=1).values * (self.gamma ** self.multi_step) * (1 - dones)

    # 计算 loss
    loss = self.loss_func(choices, targets)
    
    # 梯度下降
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
  ```

### 2.2 Double DQN
#### 2.2.1 原理
- DQN target: 
  $$
  {\rm target}(s_t) = r_t + \gamma Q_{\phi ^ -}(s_{t+1}, \argmax_{a'}Q_{\phi ^ -}(s_{t+1}, a'))
  $$
- Double DQN target: 
  $$
  {\rm target}(s_t) = r_t + \gamma Q_{\phi ^ -}(s_{t+1}, \argmax_{a'}Q_{\phi}(s_{t+1}, a'))
  $$

#### 2.2.2 实现
```python
# part of function learn of class DQN
# Double DQN 集成在类 DQN 中，只是在函数 learn 中修改了计算 target 的方法
curr_action_values: torch.Tensor = self.calc_eval_action_values(curr_states)
next_action_values: torch.Tensor = self.calc_target_action_values(next_states)

choices = curr_action_values.gather(1, curr_actions.view(self.batch_size,1)).view(self.batch_size)

# DQN
if not self.is_double:
    targets = rewards + torch.max(next_action_values,dim=1).values * (self.gamma ** self.multi_step) * (1 - dones)
# Double DQN
else:
    next_action_values_using_evalnet = self.calc_eval_action_values(next_states)
    targets = rewards + next_action_values.gather(1, torch.argmax(next_action_values_using_evalnet,dim=1).view(self.batch_size, 1)).view(self.batch_size) * (self.gamma ** self.multi_step) * (1 - dones)
```

### 2.3 Dueling DQN and Dueling DDQN
#### 2.3.1 原理
- 与 DQN 的网络直接输出 $Q$ 不同，Dueling DQN 的网络的 $Q$ 由如下公式确定：
  $$
  Q(s,a) = V(s) + A(s,a) - \frac{1}{\mathbb{A}} \sum_{a \in \mathbb{A}} A(s,a)
  $$
  ![dueling](./fig/dueling.png)

#### 2.3.2 实现
- 主要更改网络结构
  ```python
  class DuelingDQNModel(nn.Module):
    '''
    Model of Dueling DQN.
    '''
    def __init__(self, in_features, out_features):
        '''
        Init Dueling DQN model. \n
        Args: 
            in_features: dim of in_features, dim equals to the number of states for DQN
            out_features: dim of out_features, dim equals to the number of actions for DQN
        '''
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
  ```
- 更改 $Q$ 的计算方式
  ```python
  action_values = V + A - A.mean(dim=-1, keepdim=True)
  ```
