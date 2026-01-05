"""Tabular Q-learning simplu pe TwoWay4LaneEnv cu stare discretizată (lane, v-bin).

Notă: Starea folosește doar lane-ul și un bucket de viteză al ego-ului, deci
rezultatele sunt orientative. Orice modificare în ENV_CONFIG cere reantrenare.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Tuple

import numpy as np

from traffic_racer_env import make_env
from env_config import ENV_CONFIG

# --- CONSTANTE PENTRU DISCRETIZARE (V4 - Calibrate pt V_MAX = 25 m/s) ---

# 1. Viteza (m/s): Max e 25.
# Pas de 5 m/s => 6 intervale
# Bin 0: 0-5
# Bin 1: 5-10
# Bin 2: 10-15
# Bin 3: 15-20
# Bin 4: 20-25 (Viteză Maximă)
# Bin 5: >25 (Overspeed)
SPEED_BIN = 5.0       
MAX_SPEED_BINS = 6    

# 2. Distanța (m): Praguri logaritmice pentru a accentua pericolul
# Bin 0: < 15m (CRITIC - Frânează acum!)
# Bin 1: 15-30m (Pericol - Pregătește manevra)
# Bin 2: 30-60m (Atenție - Monitorizează)
# Bin 3: > 60m (Liber - Cruise)
DIST_BINS = [15, 30, 60]  

# 3. Viteza Relativă (m/s): Diferența (V_ego - V_front)
# Bin 0: < -5 (El e mai rapid, se îndepărtează - SAFE)
# Bin 1: -5 .. 5 (Viteze similare - PLUTON)
# Bin 2: > 5 (Eu sunt mai rapid, îl ajung din urmă - PERICOL)
REL_SPEED_BINS = [-5, 5]


def discretize_state(env) -> Tuple[int, int, int, int, int, int]:
    """
    Mapează starea la (lane_idx, speed_bin, dist_bin, rel_speed_bin, left_safe, right_safe).
    Total stări: 4 * 6 * 4 * 3 * 2 * 2 = 1152 stări.
    """
    
    # 1. Lane index (0-3)
    lane_idx = 0
    if hasattr(env, "vehicle") and getattr(env.vehicle, "lane_index", None):
        try:
            lane_idx = int(env.vehicle.lane_index[2])
        except Exception:
            lane_idx = 0
    lane_idx = max(0, min(int(ENV_CONFIG.get("lanes_count", 4)) - 1, lane_idx))

    # 2. Speed bin (0-5)
    speed = float(getattr(env.vehicle, "speed", 0.0))
    speed_bin = int(math.floor(speed / SPEED_BIN))
    speed_bin = max(0, min(MAX_SPEED_BINS - 1, speed_bin))

    # --- Analiză Frontală & Safety (Optimizat - O singură trecere) ---
    front_dist = 200.0
    rel_speed = 0.0
    
    # Distanțe minime pentru benzile adiacente
    left_min_d = 200.0
    right_min_d = 200.0
    
    ego_pos = env.vehicle.position[0]
    
    if hasattr(env, "road") and env.road:
        for v in env.road.vehicles:
            if v is env.vehicle: 
                continue
            
            # Extragem indexul benzii rapid (fără try-except costisitor)
            # Presupunem că vehiculele au lane_index valid ("a", "b", idx)
            v_lane_idx = v.lane_index[2]
            
            # Calcule pre-liminare
            d_raw = v.position[0] - ego_pos
            d_abs = abs(d_raw)

            # 1. Front Check (Aceeași bandă)
            if v_lane_idx == lane_idx:
                # Doar cei din față (d > 0)
                if 0 < d_raw < front_dist:
                    front_dist = d_raw
                    rel_speed = env.vehicle.speed - v.speed

            # 2. Left Safety Check (lane_idx + 1)
            elif v_lane_idx == lane_idx + 1:
                if d_abs < left_min_d:
                    left_min_d = d_abs

            # 3. Right Safety Check (lane_idx - 1)
            elif v_lane_idx == lane_idx - 1:
                if d_abs < right_min_d:
                    right_min_d = d_abs

    # 3. Distance Binning (0-3)
    dist_bin = 3 # Default: Far
    if front_dist < DIST_BINS[0]: 
        dist_bin = 0 # CRITIC
    elif front_dist < DIST_BINS[1]: 
        dist_bin = 1 # CLOSE
    elif front_dist < DIST_BINS[2]: 
        dist_bin = 2 # MEDIUM

    # 4. Relative Speed Binning (0-2)
    rs_bin = 1 # Default: Stable
    if rel_speed < REL_SPEED_BINS[0]:
        rs_bin = 0 # Pulling away
    elif rel_speed > REL_SPEED_BINS[1]:
        rs_bin = 2 # Catching up

    # 5 & 6. Dynamic Safety Checks (Left/Right)
    # Calculăm pragurile de siguranță
    # Pentru contrasens (benzile 2 și 3), buffer triplu
    
    # Left (Lane + 1)
    target_left = lane_idx + 1
    if target_left > 3:
        left_safe = 0
    else:
        is_oncoming = (target_left >= 2)
        # SCHIMBARE: Am redus multiplicatorul de la 3.0 la 1.5
        # Acum acceptă spații mai mici pe contrasens (risc asumat)
        req_dist = (10 + env.vehicle.speed * 0.5) * (1.5 if is_oncoming else 1.0)
        left_safe = 1 if left_min_d > req_dist else 0
        
    # Right (Lane - 1)
    target_right = lane_idx - 1
    if target_right < 0:
        right_safe = 0
    else:
        is_oncoming = (target_right >= 2)
        # SCHIMBARE: La fel și aici, 1.5
        req_dist = (10 + env.vehicle.speed * 0.5) * (1.5 if is_oncoming else 1.0)
        right_safe = 1 if right_min_d > req_dist else 0

    return (lane_idx, speed_bin, dist_bin, rs_bin, left_safe, right_safe)


def epsilon_greedy(Q: np.ndarray, state: Tuple[int, int, int, int, int, int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(Q.shape[-1]) # Ultima dimensiune e action space
    lane, spd, dst, rspd, lsafe, rsafe = state
    return int(np.argmax(Q[lane, spd, dst, rspd, lsafe, rsafe]))


def train_q_learning(episodes: int = 200, max_steps: int = 500, alpha: float = 0.1, gamma: float = 0.97,
                     eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995):
    env = make_env(render_mode=None)
    n_actions = env.action_space.n
    lanes = int(ENV_CONFIG.get("lanes_count", 4))

    # Q-table 7D cu dimensiunile ajustate (Speed = 6)
    Q = np.zeros((lanes, MAX_SPEED_BINS, 4, 3, 2, 2, n_actions), dtype=np.float32)

    rewards = []
    epsilon = eps_start

    print(f"Start Training... Q-Table size: {Q.size} elemente.")

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            state = discretize_state(env)
            total_reward = 0.0

            for t in range(max_steps):
                action = epsilon_greedy(Q, state, epsilon)
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = discretize_state(env)

                l, s, d, rs, ls, rsf = state
                nl, ns, nd, nrs, nls, nrsf = next_state
                
                best_next = np.max(Q[nl, ns, nd, nrs, nls, nrsf])
                td_target = reward + gamma * best_next * (0 if terminated or truncated else 1)
                td_error = td_target - Q[l, s, d, rs, ls, rsf, action]
                Q[l, s, d, rs, ls, rsf, action] += alpha * td_error

                total_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            epsilon = max(eps_end, epsilon * eps_decay)
            print(f"Ep {ep + 1:04d} | eps={epsilon:.3f} | steps={t+1} | reward={total_reward:.2f}")
            if (ep + 1) % 10 == 0:
                avg_last = np.mean(rewards[-10:])
                print(f"   >>> Avg Reward (last 10): {avg_last:.2f}")

    except KeyboardInterrupt:
        print("\n[!] Antrenament intrerupt de utilizator (CTRL+C).")
        print("Se va salva modelul cu progresul actual...")

    env.close()
    return Q, rewards


def evaluate_q(Q: np.ndarray, episodes: int = 5, max_steps: int = 500, render: bool = False):
    mode = "human" if render else None
    env = make_env(render_mode=mode)
    episode_rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        state = discretize_state(env)
        total_reward = 0.0

        if render and env.render_mode == "human":
            print(f"[Eval] Episod {ep + 1} start - deschid fereastra de randare")
            env.render() 

        for t in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon=0.0) 
            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize_state(env)
            total_reward += reward
            if render and env.render_mode == "human":
                env.render()
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"Episod {ep+1} terminat. Reward: {total_reward:.2f}")

    env.close()
    avg = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"Eval: ep_rewards={episode_rewards}, avg={avg:.2f}")
    return episode_rewards


def save_model(Q: np.ndarray, path: str):
    np.save(path, Q)
    print(f"Model salvat in {path}")


def load_model(path: str) -> np.ndarray:
    Q = np.load(path)
    print(f"Model incarcat din {path}")
    return Q


def main():
    parser = argparse.ArgumentParser(description="Tabular Q-learning demo pe TwoWay4LaneEnv")
    parser.add_argument("--episodes", type=int, default=3000, help="Număr episoade de antrenament")
    parser.add_argument("--max-steps", type=int, default=500, help="Pași pe episod (80s * 5Hz = 400)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Epsilon inițial")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Epsilon minim")
    parser.add_argument("--eps-decay", type=float, default=0.991, help="Factor de decay pe episod")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episoade de evaluare după training")
    parser.add_argument("--eval-render", action="store_true", help="Randează în evaluare (lent)")
    parser.add_argument("--model-path", type=str, default="q_table.npy", help="Cale fisier model (.npy)")
    parser.add_argument("--eval-only", action="store_true", help="Doar evaluare (fara training)")
    args = parser.parse_args()

    if args.eval_only:
        try:
            Q = load_model(args.model_path)
        except FileNotFoundError:
            print(f"Eroare: Nu am gasit modelul la {args.model_path}")
            return
    else:
        Q, rewards = train_q_learning(
            episodes=args.episodes,
            max_steps=args.max_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
        )
        print("Train terminat. Ultimele 10 reward-uri medie:", float(np.mean(rewards[-10:])))
        save_model(Q, args.model_path)

    if args.eval_episodes > 0:
        evaluate_q(Q, episodes=args.eval_episodes, max_steps=args.max_steps, render=args.eval_render)


if __name__ == "__main__":
    main()