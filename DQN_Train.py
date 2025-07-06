from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import os

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Monitor logs rewards
env = Monitor(gym.make("CartPole-v1", render_mode=None), filename=log_dir + "reward_log")

# Eval callback
eval_callback = EvalCallback(env, best_model_save_path="./models/",
                             log_path="./logs/", eval_freq=5000, deterministic=True)

model = DQN("MlpPolicy", env,
            learning_rate=5e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            verbose=1,
            tensorboard_log=log_dir)

model.learn(total_timesteps=300_000, callback=eval_callback)
model.save("models/dqn_cartpole")
env.close()
print("✅ Training complete.")
