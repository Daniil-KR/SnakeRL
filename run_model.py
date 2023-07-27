from stable_baselines3 import PPO
from snake_env import SnekEnv

models_dir = "models/1690463428"

env = SnekEnv()
env.reset()

model_path = f"{models_dir}/4000.zip"

# Загрузка весов модели
model = PPO('MlpPolicy', env, verbose=1)


# Демо
episodes = 50
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
