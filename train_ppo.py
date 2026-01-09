
import argparse
import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from traffic_racer_env import make_env

# --- Callback pentru salvarea reward-urilor ---
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file="ppo_rewards.csv", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_count = 0
        self.episode_reward = 0
        
        # Check for existing file to append
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1]
                        self.episode_count = int(last_line.split(",")[0])
            except Exception:
                pass
        else:
            with open(self.log_file, "w") as f:
                f.write("episode,reward\n")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_count += 1
                
                with open(self.log_file, "a") as f:
                    f.write(f"{self.episode_count},{ep_reward}\n")
                
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    print(f"Episod {self.episode_count}: Reward {ep_reward:.2f}")
        return True

def train(args):
    save_path = args.model_path if args.model_path else f"ppo_traffic_lr{args.learning_rate}_ent{args.ent_coef}"
    log_file = f"{save_path}_rewards.csv"
    
    print(f"--- Start Antrenament PPO ---")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Entropy Coef: {args.ent_coef}")
    print(f"Log rewards in: {log_file}")
    
    env = make_env(render_mode=None)
    env = Monitor(env) 
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"Incarc model din {args.load_model}...")
        model = PPO.load(args.load_model, env=env)
    
    callback = RewardLoggerCallback(log_file=log_file, verbose=1)
    
    model.learn(total_timesteps=args.timesteps, callback=callback)
    
    model.save(save_path)
    print(f"Model salvat in {save_path}.zip")

def evaluate(args):
    model_path = args.model_path if args.model_path else "ppo_traffic"
    # Adaugam extensia .zip daca lipseste si verificam
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
         print(f"Nu gasesc modelul {model_path}")
         return

    env = make_env(render_mode="human" if args.eval_render else None)
    model = PPO.load(model_path, env=env)

    print(f"Start evaluare ({args.eval_episodes} episoade)...")
    
    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if args.eval_render:
                env.render()
                
        print(f"Episod {ep+1}: Reward {total_reward:.2f}")

    env.close()

def main():
    parser = argparse.ArgumentParser(description="PPO cu Stable-Baselines3")
    parser.add_argument("--train", action="store_true", help="Mod antrenament")
    parser.add_argument("--eval-only", action="store_true", help="Doar evaluare")
    parser.add_argument("--timesteps", type=int, default=100000, help="Numar pasi antrenament")
    
    # Hiperparametri pentru experimente
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Rata de invatare (default: 0.0003)")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Coeficient entropie (explorare) (default: 0.0)")
    
    parser.add_argument("--eval-episodes", type=int, default=5, help="Numar episoade evaluare")
    parser.add_argument("--eval-render", action="store_true", help="Randare la evaluare")
    parser.add_argument("--model-path", type=str, default=None, help="Cale salvare model (default: auto-generat)")
    parser.add_argument("--load-model", type=str, default=None, help="Cale model de incarcat")
    
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.eval_only:
        evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
