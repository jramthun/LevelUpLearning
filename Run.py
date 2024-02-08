import retro
from RandomAgent import TimeLimitWrapper
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import time

# model = PPO.load("PPO/best_model.zip")
# model = PPO.load("PPO_4-1_best/best_model.zip")
model = PPO.load("PPO_4-1_tl=1-1/best_model.zip")

def main():
    steps = 0
    #env = retro.make(game='MegaMan2-Nes')
    print("Loading Game...")
    env = retro.make(game='SuperMarioBros-Nes', state="Level4-1")
    print("Game Loaded!")
    print("Setting up environment...")
    env = TimeLimitWrapper(env)
    env = MaxAndSkipEnv(env, 4)

    obs = env.reset()
    done = False

    for e in range(10):
        print(f"Episode {e+1} of 10")
        obs = env.reset()
        done = False
        total_reward = 0


        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
            print(total_reward, reward)
            # print(info)
            # print(total_reward, reward, info)

            time.sleep(0.01667*2)
            if done:
                obs = env.reset()
            steps += 4
            if steps % 1000 == 0:
                print(f"Total Steps: {steps}")

        print("Final Info")
        print(info)

    env.close()


if __name__ == "__main__":
    main()