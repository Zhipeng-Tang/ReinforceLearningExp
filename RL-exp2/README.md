# README

## environment
- 创建 conda 虚拟环境: 
  ```bash
  conda create -n rl python=3.9 -y
  conda activate rl
  ```
- 安装 Gymnasium: 
  ```bash
  git clone https://github.com/Farama-Foundation/Gymnasium.git
  cd Gymnasium
  python setup.py develop
  ```
- 安装 torch: 
  ```bash
  # 可能要花几分钟...
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -y
  # 如果没有 cuda
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
  # 如果没有驱动，请自行解决
  ```
- 安装 tensorboard: 
  ```bash
  pip install tensorboard
  ```
- 安装实验环境
  ```bash
  pip install gymnasium[classic-control]
  ```

## DQN
- 结合实验文档看 (RL-exp2/docu/强化学习 Lab 2.pdf)
- 尝试用神经网络去预测某个状态 state 下 (markov 过程中的状态) 采取每个动作 (action) 能获得的 reward 的期望
- 先用 gym 库中的环境产生一系列 (state, action, reward, next_state, done) 作为训练数据，然后用 DQN 算法去训练网络
