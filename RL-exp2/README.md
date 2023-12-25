# README

## environment
- create a conda env: 
  ```bash
  conda create -n rl python=3.9 -y
  conda activate rl
  ```
- install Gymnasium: 
  ```bash
  git clone https://github.com/Farama-Foundation/Gymnasium.git
  cd Gymnasium
  python setup.py develop
  ```
- install torch: 
  ```bash
  # This step may take a few minutes ...
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -y
  ```
- install tensorboard: 
  ```bash
  pip install tensorboard
  ```