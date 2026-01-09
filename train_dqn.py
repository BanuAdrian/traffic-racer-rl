
import argparse
import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_racer_env import make_env

# --- Callback pentru salvarea reward-urilor ---
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file="dqn_rewards.csv", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        # Rescrie fișierul la început
        with open(self.log_file, "w") as f:
            f.write("episode,reward\n")
        self.episode_count = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        # Adună reward-ul curent
        self.episode_reward += self.locals["rewards"][0]
        
        # Verifică dacă s-a terminat episodul
        for info in self.locals["infos"]:
            if "episode" in info:
                # SB3 calculează automat reward-ul episodului în info['episode']['r']
                # Dar pentru siguranță și consistență, folosim logica Monitor sau extragem din info
                ep_reward = info["episode"]["r"]
                self.episode_count += 1
                
                with open(self.log_file, "a") as f:
                    f.write(f"{self.episode_count},{ep_reward}\n")
                
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    print(f"Episod {self.episode_count}: Reward {ep_reward:.2f}")
                
        return True

def train(args):
    # Creăm environment-ul
    # Folosim Monitor pentru a permite SB3 să trackuiască statisticile episoadelor
    env = make_env(render_mode=None)
    env = Monitor(env) 
    
    # Modelul DQN
    # Policy: "MlpPolicy" pentru states vectoriale (nu imagini)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/"
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"Incarc model din {args.load_model}...")
        model = DQN.load(args.load_model, env=env)
    
    save_path = args.model_path if args.model_path else "dqn_traffic_racer"
    log_file = f"{save_path}_rewards.csv"
    
    print(f"Start antrenament pentru {args.timesteps} pași...")
    print(f"Log rewards in: {log_file}")
    callback = RewardLoggerCallback(log_file=log_file, verbose=1)
    
    model.learn(total_timesteps=args.timesteps, callback=callback)
    
    model.save(save_path)
    print(f"Model salvat in {save_path}.zip")

def evaluate(args):
    model_path = args.model_path if args.model_path else "dqn_traffic_racer"
    if not os.path.exists(model_path + ".zip"):
        print(f"Nu gasesc modelul {model_path}.zip")
        return

    env = make_env(render_mode="human" if args.eval_render else None)
    model = DQN.load(model_path, env=env)

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
    parser = argparse.ArgumentParser(description="DQN cu Stable-Baselines3")
    parser.add_argument("--train", action="store_true", help="Mod antrenament")
    parser.add_argument("--eval-only", action="store_true", help="Doar evaluare")
    parser.add_argument("--timesteps", type=int, default=100000, help="Numar pasi antrenament")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Numar episoade evaluare")
    parser.add_argument("--eval-render", action="store_true", help="Randare la evaluare")
    parser.add_argument("--model-path", type=str, default="dqn_traffic_racer", help="Cale salvare/incarcare model")
    parser.add_argument("--load-model", type=str, default=None, help="Cale model de incarcat pentru continuare antrenament")
    
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.eval_only:
        evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
