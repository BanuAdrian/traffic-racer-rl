"""Tabular Q-learning simplu pe TwoWay4LaneEnv cu stare discretizată."""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Tuple

import numpy as np

from traffic_racer_env import make_env
from env_config import ENV_CONFIG

# --- CONSTANTE PENTRU DISCRETIZARE (SIMPLIFICAT) ---
# State space redus: 4×3×3×2×2×2 = 288 stări (era 1152)
SPEED_BIN = 5.0       # Bin de 5 m/s: [0-5), [5-10), [10-15+]
MAX_SPEED_BINS = 3    # 3 bins: slow/medium/fast
DIST_BINS = [15, 40]  # 3 bins: close(<15m), medium(<40m), far(40m+)
REL_SPEED_APPROACHING = 2.0  # Prag: ne apropiem dacă rel_speed > 2

def discretize_state(env) -> Tuple[int, int, int, int, int, int]:
    """Mapează starea la (lane_idx, speed_bin, dist_bin, rel_speed_bin, left_safe, right_safe)."""
    
    # 1. Lane index (0-3)
    lane_idx = 0
    if hasattr(env, "vehicle") and getattr(env.vehicle, "lane_index", None):
        try: lane_idx = int(env.vehicle.lane_index[2])
        except Exception: lane_idx = 0
    lane_idx = max(0, min(int(ENV_CONFIG.get("lanes_count", 4)) - 1, lane_idx))

    # 2. Speed bin (0-5) - ajustat pentru viteze 5-15
    speed = float(getattr(env.vehicle, "speed", 0.0))
    speed_bin = int(math.floor(speed / SPEED_BIN))
    speed_bin = max(0, min(MAX_SPEED_BINS - 1, speed_bin))

    # --- Analiză Frontală & Safety ---
    front_dist = 200.0
    rel_speed = 0.0
    
    # Distanțe minime pentru benzile adiacente
    left_min_d = 200.0
    right_min_d = 200.0
    
    ego_pos = env.vehicle.position[0]
    ego_y = env.vehicle.position[1]
    
    if hasattr(env, "road") and env.road:
        for v in env.road.vehicles:
            if v is env.vehicle: continue
            
            # Folosim poziția reală X a vehiculului (nu coordonata de bandă)
            v_x = v.position[0]
            v_y = v.position[1]
            
            try:
                v_lane_idx = v.lane_index[2]
            except: continue
            
            d_raw = v_x - ego_pos  # Distanța longitudinală
            d_abs = abs(d_raw)
            
            # --- BLIND SPOT CHECK ---
            # Dacă e paralel (<10m), distanța devine 0 (pericol total)
            is_parallel = (d_abs < 10.0)

            # 1. Front Check - mașini pe aceeași bandă SAU pe benzile de contrasens dacă suntem acolo
            if v_lane_idx == lane_idx:
                if 0 < d_raw < front_dist:
                    front_dist = d_raw
                    rel_speed = env.vehicle.speed - v.speed

            # 2. Left Check (bandă cu index mai mare)
            elif v_lane_idx == lane_idx + 1:
                if d_abs < left_min_d: left_min_d = d_abs
                if is_parallel: left_min_d = 0.0

            # 3. Right Check (bandă cu index mai mic)
            elif v_lane_idx == lane_idx - 1:
                if d_abs < right_min_d: right_min_d = d_abs
                if is_parallel: right_min_d = 0.0
            
            # 4. CONTRASENS CHECK - dacă suntem pe banda 2 sau 3
            # Verificăm mașinile care vin din față pe contrasens
            if lane_idx >= 2:  # Suntem pe contrasens
                # Mașinile din contrasens vin spre noi (au lane_index[0] == "b")
                try:
                    if v.lane_index[0] == "b" and v_lane_idx == lane_idx:
                        # Mașină din contrasens pe aceeași bandă - PERICOL!
                        if v_x > ego_pos:  # E în fața noastră
                            # Distanța se reduce rapid (viteze combinate)
                            combined_approach = env.vehicle.speed + v.speed
                            effective_dist = d_raw
                            if effective_dist < front_dist:
                                front_dist = effective_dist
                                rel_speed = combined_approach  # Viteza de apropiere
                except: pass

    # 3. Distance Binning - SIMPLIFICAT la 3 bins
    dist_bin = 2  # Default: departe / liber (40m+)
    if front_dist < DIST_BINS[0]: dist_bin = 0  # PERICOL (<15m)
    elif front_dist < DIST_BINS[1]: dist_bin = 1  # Mediu (<40m)

    # 4. Relative Speed Binning - SIMPLIFICAT la 2 bins
    # 0 = nu ne apropiem (sau încet), 1 = ne apropiem rapid
    rs_bin = 1 if rel_speed > REL_SPEED_APPROACHING else 0

    # 5 & 6. Dynamic Safety Checks
    # Left - verificăm dacă putem merge pe banda din stânga
    target_left = lane_idx + 1
    if target_left > 3:
        left_safe = 0  # Nu există bandă
    else:
        is_oncoming = (target_left >= 2)
        # Contrasens cere 80m liberi, Sens normal cere 15m
        req_dist = 80.0 if is_oncoming else 15.0
        left_safe = 1 if left_min_d > req_dist else 0
        
    # Right - verificăm dacă putem merge pe banda din dreapta
    target_right = lane_idx - 1
    if target_right < 0:
        right_safe = 0  # Nu există bandă
    else:
        is_oncoming = (target_right >= 2)
        req_dist = 80.0 if is_oncoming else 15.0
        right_safe = 1 if right_min_d > req_dist else 0

    return (lane_idx, speed_bin, dist_bin, rs_bin, left_safe, right_safe)


def epsilon_greedy(Q: np.ndarray, state: Tuple[int, int, int, int, int, int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(Q.shape[-1])
    lane, spd, dst, rspd, lsafe, rsafe = state
    return int(np.argmax(Q[lane, spd, dst, rspd, lsafe, rsafe]))


def train_q_learning(episodes: int = 200, max_steps: int = 2000, alpha: float = 0.1, gamma: float = 0.97,
                     eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995,
                     Q_init: np.ndarray = None):
    """Antrenare Q-learning. max_steps mărit pentru a permite finalizarea traseului."""
    env = make_env(render_mode=None)
    n_actions = env.action_space.n
    lanes = int(ENV_CONFIG.get("lanes_count", 4))
    
    # Folosește Q-table existent sau creează unul nou
    if Q_init is not None:
        Q = Q_init
        print(f"[INFO] Q-Table încărcat: {Q.shape}")
    else:
        # State space SIMPLIFICAT: lanes(4) × speed(3) × dist(3) × rel_speed(2) × left(2) × right(2) = 288
        Q = np.zeros((lanes, MAX_SPEED_BINS, 3, 2, 2, 2, n_actions), dtype=np.float32)

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
        if render and env.render_mode == "human": env.render() 

        for t in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon=0.0) 
            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize_state(env)
            total_reward += reward
            if render and env.render_mode == "human": env.render()
            if terminated or truncated: break

        episode_rewards.append(total_reward)
        print(f"Episod {ep+1} terminat. Reward: {total_reward:.2f}")

    env.close()
    avg = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"Eval avg={avg:.2f}")
    return episode_rewards


def save_model(Q: np.ndarray, path: str):
    np.save(path, Q)
    print(f"Model salvat in {path}")


def load_model(path: str) -> np.ndarray:
    Q = np.load(path)
    print(f"Model incarcat din {path}")
    return Q


def main():
    parser = argparse.ArgumentParser(description="Tabular Q-learning")
    parser.add_argument("--episodes", type=int, default=500, help="Număr episoade")
    parser.add_argument("--max-steps", type=int, default=2000, help="Pași pe episod (mărit pentru a ajunge la final)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Epsilon inițial")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Epsilon minim")
    parser.add_argument("--eps-decay", type=float, default=0.992, help="Factor de decay")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episoade de evaluare")
    parser.add_argument("--eval-render", action="store_true", help="Randează în evaluare")
    parser.add_argument("--model-path", type=str, default="q_table.npy", help="Path model salvat")
    parser.add_argument("--load-model", type=str, default=None, help="Path model de încărcat pentru a continua antrenamentul")
    parser.add_argument("--eval-only", action="store_true", help="Doar evaluare")
    args = parser.parse_args()

    if args.eval_only:
        try: Q = load_model(args.model_path)
        except FileNotFoundError: return
    else:
        # Încarcă model existent dacă e specificat
        Q_init = None
        if args.load_model:
            try:
                Q_init = load_model(args.load_model)
                print(f"[INFO] Continuare antrenament de la {args.load_model}")
            except FileNotFoundError:
                print(f"[WARN] Modelul {args.load_model} nu există, pornesc de la zero.")
        
        Q, rewards = train_q_learning(
            episodes=args.episodes,
            max_steps=args.max_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            Q_init=Q_init,
        )
        save_model(Q, args.model_path)

    if args.eval_episodes > 0:
        evaluate_q(Q, episodes=args.eval_episodes, max_steps=args.max_steps, render=args.eval_render)

if __name__ == "__main__":
    main()