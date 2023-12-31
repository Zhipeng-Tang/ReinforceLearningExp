# README

## 环境
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
  # 如果没有驱动，sorry，请自行解决
  ```
- 安装 tensorboard: 
  ```bash
  pip install tensorboard
  ```
- 安装实验环境
  ```bash
  pip install gymnasium[classic-control]
  ```
- 安装 matplotlib
  ```bash
  pip install matplotlib
  ```

## DQN
- 结合实验文档看 (RL-exp2/docu/强化学习 Lab 2.pdf)
- 尝试用神经网络去预测某个状态 state 下 (markov 过程中的状态) 采取每个动作 (action) 能获得的 reward 的期望
- 先用 gym 库中的环境产生一系列 (state, action, reward, next_state, done) 存到 buffer 中，然后在 buffer 中随机采样，作为训练数据，最后用 DQN 算法去训练网络

## 运行
- 有奇怪的报错可以问我
- 先看看实验文档中的要求，这里选择了 CartPole-v1 和 MountainCar-v0 这两个实验环境
  - CartPole-v1: 让小车上的木棍保持平衡，并且小车不能超出屏幕范围
  - MountainCar-v0: 让小车以最小的代价从山底冲到山顶的小旗子
- 用 dqn, double_dqn, dueling_dqn, double_dueling_dqn 去学习任务的最优策略（也就是当前状态下，采取什么动作能使获得的收益最大？）
- 源码在 [RL-exp2/src/DQN.py](../RL-exp2/src/DQN.py)
- 运行: 
  ```bash
  # 一定要先进这个目录
  cd RL-exp2/src

  # 下面的命令可能需要几个小时
  # run.py 是一个训练脚本，一次训练一个任务的四种方法 (dqn, double_dqn, dueling_dqn, double_dueling_dqn)
  # -e 是环境参数，CartPole-v1 或 MountainCar-v0
  # -d 参数，这里的 gpu_id 是使用的 gpu 的编号，可以先看看哪张卡是空的，默认是 0 (其实不看也行，因为这个任务显存占用很小)
  # 可以同时训练两个任务，用不同的卡，节约时间
  # 生成的日志在 RL-exp2/src/log/CartPole-v1
  python run.py -e CartPole-v1 -d gpu_id
  # 可以用 tensorboard 实时看一下运行情况
  tensorboard --logdir log/CartPole-v1

  # 这个时间应该稍微短点
  # 生成的日志在 RL-exp2/src/log/MountainCar-v0
  python run.py -e MountainCar-v0 -d gpu_id
  tensorboard --logdir log/MountainCar-v0
  ```
- 画 reward 曲线: 
  ```bash
  # 也要在这个目录下执行 RL-exp2/src
  # -d 参数是要画的曲线的日志的目录
  # -s 是平滑参数，0 到 1 之间，越大越平滑，默认是 0.99。这里用 0.99 是因为曲线波动大的吓人，你也可以试试更小的平滑参数，看看能不能看的过去
  # -o 是输出文件的路径，你也可以改成你想要的名字
  python plot.py -d log/CartPole-v1 -s 0.99 -o CartPole-v1.png

  python plot.py -d log/MountainCar-v0 -s 0.99 -o MountainCar-v0.png
  ```
- 录制视频: 
  ```bash
  # 也要在这个目录下执行 RL-exp2/src
  # 注意: 一定要在有图形化界面的机器上运行下面的命令，因为要录屏。如果训练用的服务器，可以把生成的 RL-exp2/src/log 目录复制到你自己的机器
  # git 并没有管理 RL-exp2/src/log 这个目录
  # 建议: 如果你用的是 windows，可以用 windows 自带的截图工具录屏 + 自带的 Microsoft Clipchamp 剪辑，或者你有更高级的工具
  # 建议: 如果你用的是 ubuntu，可以用 kazam? 我不会！

  # 下面每条命令都会渲染出相应方法的学习出的最优策略的视频，默认会重复 2000 次，记得录屏！
  # dqn
  python DQN.py -t -e CartPole-v1 -m dqn --ckpt log/CartPole-v1/dqn/ckpt/final.pth
  # double dqn
  python DQN.py -t -e CartPole-v1 -m dqn --double --ckpt log/CartPole-v1/double_dqn/ckpt/final.pth
  # dueling
  python DQN.py -t -e CartPole-v1 -m dueling --ckpt log/CartPole-v1/dueling/ckpt/final.pth
  # double dueling
  python DQN.py -t -e CartPole-v1 -m dueling --double --ckpt log/CartPole-v1/double_dueling/ckpt/final.pth

  # dqn
  python DQN.py -t -e MountainCar-v0 -m dqn --ckpt log/MountainCar-v0/dqn/ckpt/final.pth
  # double dqn
  python DQN.py -t -e MountainCar-v0 -m dqn --double --ckpt log/MountainCar-v0/double_dqn/ckpt/final.pth
  # dueling
  python DQN.py -t -e MountainCar-v0 -m dueling --ckpt log/MountainCar-v0/dueling/ckpt/final.pth
  # double dueling
  python DQN.py -t -e MountainCar-v0 -m dueling --double --ckpt log/MountainCar-v0/double_dueling/ckpt/final.pth
  ```
**必做部分完工！！！**
必做部分的源码参见[必做](./src/DQN_neccessary.py)

## DQN 扩展
### multi_step learning
- 就是训练数据不只是包含 (state, action, reward, next_state, done)，而是连续的几个 (state, action, reward, next_state, done)
- 运行：
  ```bash
  # 也要在这个目录下执行 RL-exp2/src
  # CUDA_VISIBLE_DEVICES 指定用到的 gpu，gpu_id 是你要用的 gpu 的编号，跟前面一样，最好是空闲的
  # 这里只尝试了 MountainCar-v0 环境下的 multi_step=3,5，你也可以尝试更多环境或步长
  # 普通的 dqn 相当于 multi_step=1
  # 生成的日志在 RL-exp2/src/log/multi_step/MountainCar-v0
  CUDA_VISIBLE_DEVICES=gpu_id python DQN.py -e MountainCar-v0 -m dqn --multi_step 3 --logdir log/multi_step
  CUDA_VISIBLE_DEVICES=gpu_id python DQN.py -e MountainCar-v0 -m dqn --multi_step 5 --logdir log/multi_step

  # 把之前的普通的 dqn 的日志复制一份到 multi_step 的日志里，方便画图比较
  cp -r log/MountainCar-v0/dqn log/multi_step/MountainCar-v0/
  ```
- 画 reward 曲线
  ```bash
  python plot.py -d log/multi_step/MountainCar-v0 -s 0.99 -o MountainCar-v0_multi_step.png
  ```
- 如果你想的话，也可以录制视频
  ```bash
  # 同样的配方，同样的操作，记得录屏！
  python DQN.py -t -e MountainCar-v0 -m dqn --multi_step 3 --ckpt log/multi_step/MountainCar-v0/multi_step_3_dqn/ckpt/final.pth

  python DQN.py -t -e MountainCar-v0 -m dqn --multi_step 5 --ckpt log/multi_step/MountainCar-v0/multi_step_5_dqn/ckpt/final.pth
  ```

**后面还有 noisy-net, prioritized_relay, categorical_dqn 有时间再更新！**