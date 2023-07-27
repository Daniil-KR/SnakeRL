from stable_baselines3 import PPO
import os
from snake_env import SnekEnv
import time


# Директории для хранения логов и весов модели
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# Создание окружения для обучения
env = SnekEnv()
env.reset()

# Выбор алгоритма для обучения - PPO
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# Процесс обучения
TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")