## Base Packages
import gym
import retro
import numpy as np
import os
import psutil
from typing import Callable

## Stable Baselines
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

## Custom Wrapper
class TimeLimitWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  :param max_steps: (int) Max number of steps per episode
  """
  def __init__(self, env, max_steps=10000):
    # Call the parent constructor, so we can access self.env later
    super(TimeLimitWrapper, self).__init__(env)
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
  
  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    self.current_step = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    # Overwrite the done signal when 
    if self.current_step >= self.max_steps:
      done = True
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    info['Current_Step'] = self.current_step
    return obs, reward, done, info

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


def make_env(env_id, rank, seed=0, state=retro.State.DEFAULT, max_steps=2000):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param state: (retro.State) state file used to initialize the environment
    """
    def _init():
        #env = gym.make(env_id)
        env = retro.make(game=env_id, state=state)
        env = TimeLimitWrapper(env, max_steps=max_steps)
        env = MaxAndSkipEnv(env, 4) # keep only every fourth frame
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def piecewise_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Piecewise schedule
    :param initial_value: (float) Initial value
    :return: (function) Piecewise schedule
    """
    def value_schedule(progress: float) -> float:
        if progress < 0.1:
            return initial_value * 0.01
        elif progress < 0.5:
            return initial_value * 0.1
        else:
            return initial_value
    
    return value_schedule

def human_format(num, ends=["", "K", "M", "B", "T"]):
    return ends[int(np.floor(np.log10(num))/3)]

# Preconfigured Implementation
def train_PPO(env_id: str = None, num_cpu: int = np.floor(psutil.cpu_count(logical=False)*5/6), log_dir: str = "./model_logs", tb_log_dir: str = "./tb_logs", lr: float = 3e-5, state=retro.State.DEFAULT, verbose: int = 1, max_timesteps: int = 1000000, max_epoch_steps: int = 4500, pretrained: str = None):
    if isinstance(state, retro.State):
        if state == retro.State.DEFAULT:
            state = "Default"
        elif state == retro.State.RANDOM:
            state = "Random"
        elif state == retro.State.NONE:
            raise Exception("Please set the state to a supported value or leave blank")
        else:
            raise ValueError(f"Unknown state: {state}")

    time_step_prefix = int(max_timesteps / 10**(np.log10(max_timesteps) // 3 * 3))
    log_name = f"PPO_{time_step_prefix}{human_format(max_timesteps)}_lr={np.format_float_scientific(lr, precision=2, trim='-')}_state={state}_limit={max_epoch_steps}_pretrained="

    if log_dir is None:
        log_dir = "./PPO"

    log_dir = f"{log_dir}/{env_id}/{log_name}/"
    os.makedirs(log_dir, exist_ok=True)

    env = VecMonitor(SubprocVecEnv([make_env(env_id, i, state=state, max_steps=max_epoch_steps) for i in range(num_cpu)]),f"{log_dir}/TestMonitor")

    if pretrained is not None:
        custom_objects = { 'learning_rate': piecewise_schedule(lr), 'seed': 0 }
        model = PPO.load(pretrained, env=env, custom_objects=custom_objects)
    else:
        model = PPO("CnnPolicy", env, verbose=verbose, tenorboard_log=f"{tb_log_dir}/{env_id}", learning_Rate=lr)

    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
    model.learn(total_timesteps=max_timesteps, callback=callback, tb_log_name=log_name)
    model.save(env_id)
    print("------------- Done Learning -------------")

if __name__ == '__main__':
    # Use preconfigured training - Comment this out for manual configurations
    # train_PPO(env_id="SuperMarioBros-Nes", num_cpu=10, lr=3e-5, state="Level8-1", max_timesteps=3000000, max_epoch_steps=3000, pretrained=None)

    # Set each value manually - Uncomment below this line
    env_id = "SuperMarioBros-Nes" # Name of the ROM loaded (not provided for legal reasons)
    num_cpu = 10

    # Create log dir
    log_dir = "PPO_2-1_tl/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create the vectorized environment
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i, state="Level2-1", max_steps=3000) for i in range(num_cpu)]),f"{log_dir}TestMonitor")

    custom_objects = { 'learning_rate': piecewise_schedule(3e-5), 'seed': 0 }
    model = PPO.load("./PPO_1-1_best/best_model.zip", env=env, custom_objects=custom_objects)
    # policy = model.policy

    # policy_kwargs = dict(net_arch=dict(vf=[64, 64, 64], pi=[64, 64, 64]))
    # model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./board/", learning_rate=3e-5, seed=0)
    # model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./board/", learning_rate=piecewise_schedule(3e-5), seed=0)

    # Required for transfer learning
    # custom_objects = { 'learning_rate': piecewise_schedule(3e-6), 'seed': 0 }
    # model = PPO.load("./PPO_1-1_best/best_model.zip", env=env, custom_objects=custom_objects)

    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
    model.learn(total_timesteps=3000000, callback=callback, tb_log_name="PPO_3M_lr-sched=3e-5_state=2-1_limit=3000_skip=4_custom-reward=lives-1_seed=0_tl")
    model.save(env_id)
    print("------------- Done Learning -------------")