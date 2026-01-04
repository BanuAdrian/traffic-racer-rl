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

# Discretizare grosieră pentru viteză (m/s) în 10 intervale a câte 5 m/s.
SPEED_BIN = 5.0
MAX_SPEED_BINS = 10
# Discretizare distanță față (m): 0=aproape, 1=mediu, 2=departe/liber
DIST_BINS = [20, 50]  # praguri


def discretize_state(env) -> Tuple[int, int, int, int, int]:
    """Mapează starea la (lane_idx, speed_bin, dist_bin, left_safe, right_safe)."""
    
    # 1. Lane index
    lane_idx = 0
    if hasattr(env, "vehicle") and getattr(env.vehicle, "lane_index", None):
        try:
            lane_idx = int(env.vehicle.lane_index[2])
        except Exception:
            lane_idx = 0
    lane_idx = max(0, min(int(ENV_CONFIG.get("lanes_count", 4)) - 1, lane_idx))

    # 2. Speed bin
    speed = float(getattr(env.vehicle, "speed", 0.0))
    speed_bin = int(math.floor(speed / SPEED_BIN))
    speed_bin = max(0, min(MAX_SPEED_BINS - 1, speed_bin))

    # Helper pentru distanțe
    def get_lane_dist(target_lane_idx):
        # Verificăm vehicule pe banda target_lane_idx (a->b)
        # DAR și pe banda suprapusă (b->a) dacă există.
        # Mapare simplificată:
        # Lane 0 (a->b) <-> None
        # Lane 1 (a->b) <-> None
        # Lane 2 (a->b) <-> Lane 0 (b->a)
        # Lane 3 (a->b) <-> Lane 1 (b->a)
        
        min_d = 200.0
        ego_pos = env.vehicle.position[0]
        
        if not hasattr(env, "road") or not env.road:
            return min_d

        for v in env.road.vehicles:
            if v is env.vehicle:
                continue
            
            v_lane_idx = v.lane_index[2]
            v_lane_from = v.lane_index[0]
            v_lane_to = v.lane_index[1]
            
            # Verificăm dacă vehiculul e pe banda target
            is_target = False
            
            # Cazul 1: Vehicul pe sensul nostru (a->b)
            if v_lane_from == "a" and v_lane_to == "b" and v_lane_idx == target_lane_idx:
                is_target = True
            
            # Cazul 2: Vehicul pe sens opus (b->a) care se suprapune
            # Lane 2 (a->b) se suprapune cu Lane 0 (b->a)
            if target_lane_idx == 2 and v_lane_from == "b" and v_lane_to == "a" and v_lane_idx == 0:
                is_target = True
            # Lane 3 (a->b) se suprapune cu Lane 1 (b->a)
            if target_lane_idx == 3 and v_lane_from == "b" and v_lane_to == "a" and v_lane_idx == 1:
                is_target = True
                
            if is_target:
                # Distanța absolută (față sau spate) pentru siguranță la schimbare
                d = abs(v.position[0] - ego_pos)
                if d < min_d:
                    min_d = d
        return min_d

    # 3. Front distance (pe banda curentă)
    # Aici ne interesează doar ce e în FAȚĂ (>0)
    front_dist = 200.0
    ego_pos = env.vehicle.position[0]
    if hasattr(env, "road") and env.road:
        for v in env.road.vehicles:
            if v is env.vehicle:
                continue
            # Check same lane logic as above but only forward
            # Simplificare: verificăm doar dacă e exact pe lane_index-ul nostru
            if v.lane_index == env.vehicle.lane_index:
                d = v.position[0] - ego_pos
                if 0 < d < front_dist:
                    front_dist = d
    
    dist_bin = 0
    if front_dist < DIST_BINS[0]:
        dist_bin = 0
    elif front_dist < DIST_BINS[1]:
        dist_bin = 1
    else:
        dist_bin = 2

    # 4. Left Safe (lane_idx + 1)
    left_safe = 0
    if lane_idx < 3: # Putem merge stânga
        d = get_lane_dist(lane_idx + 1)
        if d > 15.0: # E loc să intrăm
            left_safe = 1
    
    # 5. Right Safe (lane_idx - 1)
    right_safe = 0
    if lane_idx > 0: # Putem merge dreapta
        d = get_lane_dist(lane_idx - 1)
        if d > 15.0:
            right_safe = 1

    return lane_idx, speed_bin, dist_bin, left_safe, right_safe


def epsilon_greedy(Q: np.ndarray, state: Tuple[int, int, int, int, int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(Q.shape[5])
    lane, spd, dst, lsafe, rsafe = state
    return int(np.argmax(Q[lane, spd, dst, lsafe, rsafe]))


def train_q_learning(episodes: int = 200, max_steps: int = 500, alpha: float = 0.1, gamma: float = 0.97,
                     eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995):
    env = make_env(render_mode=None)
    n_actions = env.action_space.n
    lanes = int(ENV_CONFIG.get("lanes_count", 4))

    # Q-table 6D: [lane, speed, dist, left_safe, right_safe, action]
    Q = np.zeros((lanes, MAX_SPEED_BINS, 3, 2, 2, n_actions), dtype=np.float32)

    rewards = []
    epsilon = eps_start

    for ep in range(episodes):
        obs, info = env.reset()
        state = discretize_state(env)
        total_reward = 0.0

        for t in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(env)

            lane, spd, dst, lsafe, rsafe = state
            nlane, nspd, ndst, nlsafe, nrsafe = next_state
            best_next = np.max(Q[nlane, nspd, ndst, nlsafe, nrsafe])
            td_target = reward + gamma * best_next * (0 if terminated or truncated else 1)
            td_error = td_target - Q[lane, spd, dst, lsafe, rsafe, action]
            Q[lane, spd, dst, lsafe, rsafe, action] += alpha * td_error

            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(eps_end, epsilon * eps_decay)
        print(f"Ep {ep + 1:04d} | eps={epsilon:.3f} | reward={total_reward:.2f}")
        if (ep + 1) % 10 == 0:
            avg_last = np.mean(rewards[-10:])
            print(f"   >>> Avg Reward (last 10): {avg_last:.2f}")

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
            env.render()  # forțează inițializarea viewer-ului

        for t in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon=0.0)  # greedy
            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize_state(env)
            total_reward += reward
            if render and env.render_mode == "human":
                env.render()
                time.sleep(0.1)
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

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
    parser.add_argument("--max-steps", type=int, default=500, help="Pași pe episod")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Epsilon inițial")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Epsilon minim")
    parser.add_argument("--eps-decay", type=float, default=0.999, help="Factor de decay pe episod")
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
