import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN, PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from traffic_racer_env import make_env
# Importăm funcția de discretizare pentru Q-Learning
try:
    from q_learning import discretize_state
except ImportError:
    print("Eroare: Nu am putut importa 'discretize_state' din q_learning.py.")
    print("Asigură-te că q_learning.py este în același folder.")
    exit()

# --- CONFIGURARE ---
EPISODES = 20  # Număr episoade de test per agent
MAX_STEPS = 2000 # Limita de pași (trebuie să fie suficientă pt a termina cursa)
MODEL_PATHS = {
    "Q-Learning": "q_table_final_v4.npy",     # Calea către tabela Q
    "DQN": "dqn_traffic_racer_v1",        # Calea către modelul DQN (SB3 adauga .zip)
    "PPO": "ppo_traffic_racer_v1.0"         # Calea către modelul PPO (am adaugat .zip manual)
}

def evaluate_agent(agent_name, model, env, n_episodes):
    """Rulează bucla de evaluare pentru un singur agent."""
    results = {
        "rewards": [],
        "steps": [],
        "success": [],
        "collisions": []
    }
    
    print(f"\nEvaluating {agent_name}...")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        # Q-Learning are nevoie de starea discretizată
        if agent_name == "Q-Learning":
            state = discretize_state(env)
            
        while not (terminated or truncated):
            # 1. SELECTARE ACȚIUNE
            if agent_name == "Q-Learning":
                # Greedy action (argmax)
                # State: (lane_idx, speed_bin, sit_left, sit_center, sit_right)
                lane, spd, sl, sc, sr = state
                action = int(np.argmax(model[lane, spd, sl, sc, sr]))
            else:
                # SB3 Models (DQN, PPO)
                # deterministic=True înseamnă că nu mai explorează, ci alege ce știe mai bine
                action, _ = model.predict(obs, deterministic=True)
            
            # 2. PAS ÎN MEDIU
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update stare pt Q-Learning
            if agent_name == "Q-Learning":
                state = discretize_state(env)
                
            total_reward += reward
            step_count += 1
            
            if step_count >= MAX_STEPS:
                truncated = True

        # 3. STATISTICI FINALE EPISOD
        results["rewards"].append(total_reward)
        results["steps"].append(step_count)
        
        # Verificăm succesul (dacă a ajuns la final)
        # Ne bazăm pe reward-ul mare de final (+50) sau poziția mașinii
        is_success = False
        if env.unwrapped.vehicle.position[0] >= env.unwrapped.config["road_length"] - 20:
            is_success = True
            
        results["success"].append(1 if is_success else 0)
        results["collisions"].append(1 if env.unwrapped.vehicle.crashed else 0)
        
        print(f"  Ep {ep+1}: Reward={total_reward:.1f}, Steps={step_count}, Success={'YES' if is_success else 'NO'}")
        
    return results

def main():
    print("--- START EVALUARE ---")
    # Creăm mediul (fără render pentru viteză, sau 'human' dacă vrei să vezi)
    try:
        env = make_env(render_mode=None)
        print("Mediu creat cu succes.")
    except Exception as e:
        print(f"CRITICAL: Nu pot crea mediul: {e}")
        return
    
    all_metrics = []

    print(f"Modele de testat: {list(MODEL_PATHS.keys())}")

    # --- 1. EVALUARE Q-LEARNING ---
    if "Q-Learning" in MODEL_PATHS:
        path = MODEL_PATHS["Q-Learning"]
        if os.path.exists(path):
            try:
                q_table = np.load(path)
                stats = evaluate_agent("Q-Learning", q_table, env, EPISODES)
                for r, s, suc in zip(stats["rewards"], stats["steps"], stats["success"]):
                    all_metrics.append({"Agent": "Q-Learning", "Reward": r, "Steps": s, "Success": suc})
            except Exception as e:
                print(f"Eroare la incarcarea Q-Learning: {e}")
        else:
            print(f"Skip Q-Learning: Fisierul '{path}' nu exista.")

    # --- 2. EVALUARE DQN ---
    if "DQN" in MODEL_PATHS:
        path = MODEL_PATHS["DQN"]
        # SB3 adauga automat .zip la load, deci verificam daca exista path sau path.zip
        check_path = path if path.endswith(".zip") else path + ".zip"
        if os.path.exists(check_path):
            try:
                model = DQN.load(path, env=env)
                stats = evaluate_agent("DQN", model, env, EPISODES)
                for r, s, suc in zip(stats["rewards"], stats["steps"], stats["success"]):
                    all_metrics.append({"Agent": "DQN", "Reward": r, "Steps": s, "Success": suc})
            except Exception as e:
                print(f"Eroare la incarcarea DQN: {e}")
        else:
            print(f"Skip DQN: Fisierul '{check_path}' nu exista.")

    # --- 3. EVALUARE PPO ---
    if "PPO" in MODEL_PATHS:
        path = MODEL_PATHS["PPO"]
        check_path = path if path.endswith(".zip") else path + ".zip"
        if os.path.exists(check_path):
            try:
                model = PPO.load(path, env=env)
                stats = evaluate_agent("PPO", model, env, EPISODES)
                for r, s, suc in zip(stats["rewards"], stats["steps"], stats["success"]):
                    all_metrics.append({"Agent": "PPO", "Reward": r, "Steps": s, "Success": suc})
            except Exception as e:
                print(f"Eroare la incarcarea PPO: {e}")
        else:
            print(f"Skip PPO: Fisierul '{check_path}' nu exista.")
    
    env.close()

    # --- 4. VIZUALIZARE & RAPORT ---
    df = pd.DataFrame(all_metrics)
    
    if df.empty:
        print("Nu am date pentru generarea graficelor.")
        return

    # Print Tabel Sumar
    summary = df.groupby("Agent").agg(
        Avg_Reward=("Reward", "mean"),
        Std_Reward=("Reward", "std"),
        Success_Rate=("Success", "mean"),
        Avg_Steps=("Steps", "mean")
    )
    # Convertim rata de succes în procent
    summary["Success_Rate"] *= 100
    
    print("\n=== REZULTATE COMPARATIVE FINALE ===")
    print(summary)
    
    # Salvare CSV pentru raport
    summary.to_csv("rezultate_comparative.csv")
    
    # --- Generare Grafice ---
    sns.set_theme(style="whitegrid")
    
    # Fig 1: Boxplot Rewards (Arată distribuția și stabilitatea)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Agent", y="Reward", data=df, palette="viridis")
    plt.title(f"Distribuția Reward-urilor pe {EPISODES} Episoade de Test")
    plt.ylabel("Reward Total")
    plt.savefig("comparatie_rewards_boxplot.png")
    plt.show()
    
    # Fig 2: Barplot Success Rate
    plt.figure(figsize=(8, 5))
    # Agregăm datele pentru barplot
    success_df = df.groupby("Agent")["Success"].mean().reset_index()
    success_df["Success"] *= 100
    
    sns.barplot(x="Agent", y="Success", data=success_df, palette="viridis")
    plt.title(f"Rata de Succes (Ajunge la Final) - {EPISODES} Episoade")
    plt.ylabel("Success Rate (%)")
    plt.savefig("comparatie_success_rate.png")
    plt.show()

if __name__ == "__main__":
    main()