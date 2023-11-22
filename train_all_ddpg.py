import os
import time
from tqdm.auto import tqdm
import gymnasium as gym
from stable_baselines3 import DDPG
from gym_tictactoe.env import TicTacToeEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

tmp_path = "logs/"
# set up logger

# os.mkdir(tmp_path, if_exists=True)


for size in tqdm(range(3, 5)):
    print("\n\n==============TRAINING SIZE {}================".format(size))
    env = TicTacToeEnv(size=size)
    eval_callback = EvalCallback(env, best_model_save_path=f"ddpgmodel_{size}/",
                                 log_path=f"logs/{size}", eval_freq=500,
                                 deterministic=True, render=False, n_eval_episodes=10)

    new_logger = configure(tmp_path+f"{size}", ["stdout", "csv", "tensorboard"])
    model = DDPG("MlpPolicy", env, verbose=1, batch_size=300)
    
    
    # os.mkdir(tmp_path+f"{size}", if_exists=True)
    model.set_logger(new_logger)
    start = time.time()
    model.learn(total_timesteps=5000, log_interval=10, callback=[eval_callback], progress_bar=True,)
    print(f"Training took {time.time()-start} seconds")
    model.save(f"ddpgmodel_{size}")
    
    # os.mkdir(tmp_path+f"{size}", if_exists=True)

    # model.set_logger(new_logger)
    # model.learn(total_timesteps=2000, log_interval=10, callback=eval_callback, progress_bar=True,)
    # model.save(f"ddpgmodel_{size}")
    
    del model # remove to demonstrate saving and loading

    model = DDPG.load(f"ddpgmodel_{size}/best_model.zip")
    start = time.time()
    done = False
    obs, info = env.reset()
    observation = obs
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        print("Action: ", action)
        observation, reward, done, trunc, _ = env.step(action)
        print(observation, reward, done)
        env.render()

        if done or trunc:
            print("Game Over!")
            break
    
    
    print(f"Testing took {time.time()-start} seconds")