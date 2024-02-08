# Level-Up Learning: Transfer Learning-based Policy Generalization in Platforming Video Games
>  ECE 595 Reinforcement Learning, Fall 2023 Semester Project by Joe Kawiecki and John Ramthun

## Table of Contents
1. [Getting Started](#getting-started)
2. [Usage](#usage)
    1. [Training](#training)
        1. [Preconfigured](#using-the-preconfigured-training-approach)
        1. [Manual](#manually-configuring-training)
    1. [Inferencing](#inferencing)
3. [Authors and acknowledgment](#authors-and-acknowledgements)

## Getting Started
We used conda as our package manager to allow easier management of package versions. Please be aware that significant version incompatibilites have been seen during the creation of this environment. Please use the included `environment.yml` or `requirements.txt` to reproduce the environment with the necessary packages.

In the event of issues creating the environment, ensure that the following versions are met:
- `stable-baselines3==1.8.0`
- `gym-retro==0.8.0`
- `gym==0.21.0`
- `numpy=1.24.4`

These should minimize the number of issues faced when reproducing our environment.

This environment was created with the following PC in mind:
- Windows 11 (supports WSL2)
- RTX 30-Series (which may determine the CUDA packages required/available)
- AMD Ryzen CPU (which may affect C library implementation), YMMV

You will also need to obtain your own copy of the NES Super Mario Bros (USA) ROM. For legal reasons, we cannot provide a copy.

To import a new game after installing Gym Retro, please run `python3 -m retro.import /path/to/your/ROMs/directory/` and all valid ROMs will be installed. See https://retro.readthedocs.io/en/latest/getting_started.html for complete instructions.

We also modified core files of Gym Retro. Please replace the default `retro/retro_env.py`, `retro/enums.py`, `retro/data/stable/SuperMarioBros-Nes/data.json`, and `retro/data/stable/SuperMarioBros-Nes/scenario.json` with our provided versions of the files to implement our custom reward and random stage selection.

## Usage
We offer no command line implementation. Please open and run the code using your preferred editor/IDE

### Training
Training is handled in `Train.py`

NOTE: The code must remain under the line `if __name__ == '__main__':` to maintain multiprocessing support. Unexpected behavior will occur otherwise.

#### Using the preconfigured training approach
1. `env_id` must match the name provided upon import. `SuperMarioBros-Nes` is the assumed game, but others should work
1. Set `num_cpu` to your preferred amount. If left blank, we use 5/6 of the available PHYSICAL CPU cores for best overall performance. Please note that fewer cores increases training time
1. Set your preferred base `log_dir`. We will checkpoint the model in this folder.
1. Set your preferred learning rate. Default: 3e-5
1. Set your preferred `state` (ref: game save state). Must be valid for the ROM. Super Mario Bros assumes states in the format `LevelX-1` where X is the world number (1-8). No additional save states are provided. Two other accepted states are permitted:
    - `retro.State.DEFAULT`: alias of `Level1-1`
    - `retro.State.RANDOM`: custom function that allows retro to pick a new state upon each env reset (or episode end)
1. Set your preferred maximum number of timesteps. Default: 1 million. On our system, training progresses at approximately 1M steps / hour.
1. Set `max_epoch_steps`. Default: 3000. We limit the amount of time the agent can explore per epoch to save time while training. Training has revealed that the agent rarely meets this threshold.
1. If using a pretrained mode, set `pretrained` to the PATH of the saved `.zip`. Else, set to `NONE` to train from scratch.

#### Manually configuring training
We also allow users to configure the training procedure at the end of the file. Please follow the above guidelines to configure the training process and note that we provide no additional safeguards. Please comment out the call to `train_PPO` and uncomment all relevant code below it.

### Inferencing
Inferencing is handled in `Run.py`

1. Make sure to provide the correct model to `PPO.load(path)`
1. Set the desired `state` to `retro.make()`. The formatting is the same as the above.
1. The render speed can be adjusted by modifying `time.sleep(0.1667*C)`, where C is the fast forward coefficient. When C=1, the output is every fourth frame at 60fps, which is visually unappealing. Instead, setting C to a multiple of 2 makes the render smoother.

Everything else can be left alone. The true reward and cummulative true reward will be printed in the console after each time step.

## Authors and Acknowledgements
Implementation and modifications by [John Ramthun](https://github.com/jramthun)

This work is based on the work of [ClarityCoders](https://github.com/ClarityCoders). The original implementation can be found at https://github.com/ClarityCoders/MarioPPO. We aimed to make their more user friendly and customizable.