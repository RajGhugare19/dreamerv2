# Model-Based Active Exploration (MAX)
Code for reproducing experiments in [Model-Based Active Exploration](https://arxiv.org/abs/1810.12162
), ICML 2019 

Written in PyTorch v1.0. 

Code relies on [sacred](http://sacred.readthedocs.io) for managing experiments and hyper-parameters.

### Overview:
* `envs/`: contains the environments used.
* `main.py`: contains the main algorithm and baselines through modes.
* `models.py`: a fast parallel implementation of an ensemble of models which can are trained with negative log-likelihood loss.
* `utilities.py`: contains the all the utilities (exploration objectives) used in the paper.
* `imagination.py`: contains code that constructs a virtual MDP using the model ensemble.
* `sac.py`: contains a simple Soft Actor-Critic implementation.
* `sacred_fetcher.py`: script to download experiment artifacts stored in MongoDB.

### Installation
* Install required dependencies:

    ```bash
    sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    ```

* Create conda environment with required dependencies:

    ```bash
    conda env create -f conda_env.yml
    ```

* Download and setup MuJoCo binaries. The project uses `mujoco` and `mujoco_py` version 1.50. 

    ```bash
    mkdir ~/.mujoco/
    cd .mujoco/
    wget -c https://www.roboti.us/download/mjpro150_linux.zip
    unzip mjpro150_linux.zip
    rm mjpro150_linux.zip
    ```

    Obtain MuJoCo license key and place it `.mujoco/` directory created above with filename `mjkey.txt`.

* Append the following to `~/.bashrc`:

    ```bash
    # MuJoCo
    export LD_LIBRARY_PATH=:/home/<USER>/.mujoco/mjpro150/bin
    
    if [ -f /usr/lib/x86_64-linux-gnu/libGLEW.so ]; then    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<USER>/.mujoco/mjpro150/bin:/usr/lib/nvidia-390
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-375
    fi

    ```

* Quick test of MuJoCo installation

    ```python
    >>> import gym
    >>> gym.make('HalfCheetah-v2')
    ```

### Commands
Execute the commands listed below from the code directory to reproduce the results.

#### Half Cheetah
* MAX:
```
python main.py with max_explore env_noise_stdev=0.02
```

* Trajectory Variance Active Exploration:
```
python main.py with max_explore utility_measure=traj_stdev policy_explore_alpha=0.2 env_noise_stdev=0.02
```

* Renyi Divergence Reactive Exploration:
```
python main.py with max_explore exploration_mode=reactive env_noise_stdev=0.02
```

* Prediction Error Reactive Exploration:
```
python main.py with max_explore exploration_mode=reactive utility_measure=pred_err policy_explore_alpha=0.2 env_noise_stdev=0.02
```

* Random Exploration:
```
python main.py with random_explore env_noise_stdev=0.02
```



#### Ant

* MAX:
```
python main.py with max_explore env_name=MagellanAnt-v2 env_noise_stdev=0.02 eval_freq=1500 checkpoint_frequency=1500 ant_coverage=True
```

* Trajectory Variance Active Exploration:
```
python main.py with max_explore env_name=MagellanAnt-v2 utility_measure=traj_stdev policy_explore_alpha=0.2 env_noise_stdev=0.02 eval_freq=1500 checkpoint_frequency=1500 ant_coverage=True
```

* Renyi Divergence Reactive Exploration:
```
python main.py with max_explore env_name=MagellanAnt-v2 exploration_mode=reactive env_noise_stdev=0.02 eval_freq=1500 checkpoint_frequency=1500 ant_coverage=True
```

* Prediction Error Reactive Exploration:
```
python main.py with max_explore env_name=MagellanAnt-v2 exploration_mode=reactive utility_measure=pred_err policy_explore_alpha=0.2 env_noise_stdev=0.02 eval_freq=1500 checkpoint_frequency=1500 ant_coverage=True
```

* Random Exploration:
```
python main.py with random_explore env_name=MagellanAnt-v2 env_noise_stdev=0.02 eval_freq=1500 checkpoint_frequency=1500 ant_coverage=True
```

## Magellan
Magellan is the internal code name of the project inspired by life of [Ferdinand Magellan](https://en.wikipedia.org/wiki/Ferdinand_Magellan).
