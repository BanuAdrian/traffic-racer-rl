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

# --- CONSTANTE PENTRU DISCRETIZARE ---
SPEED_BIN = 5.0
MAX_SPEED_BINS = 3

# Situations
SIT_SAFE = 0
SIT_CAUTION = 1
SIT_DANGER = 2
NUM_SITUATIONS = 3

# Distances for situations
DIST_DANGER_SAME = 30.0
DIST_CAUTION_SAME = 60.0
DIST_DANGER_ONCOMING = 60.0
DIST_CAUTION_ONCOMING = 100.0

LANE_WIDTH = 4.0

def get_lane_situation(env, lane_idx_check: int, ego_pos: np.ndarray, ego_speed: float, ego_lane: int) -> int:
    """
    Calculează situația (SAFE, CAUTION, DANGER) pentru o bandă specifică.
    Ia în considerare distanța și viteza relativă față de cea mai apropiată mașină din față.
    """
    closest_dist = 200.0
    closest_v = None
    
    # Identificăm sensul benzii verificate
    # 0, 1 -> Sens A (același cu ego start)
    # 2, 3 -> Sens B (contrasens)
    is_oncoming_lane = (lane_idx_check >= 2)
    
    if hasattr(env, "road") and env.road:
        for v in env.road.vehicles:
            if v is env.vehicle:
                continue
                
            # Verificăm dacă vehiculul e pe banda pe care o analizăm
            # Structura lane_index: ("a", "b", 0) sau ("b", "a", 0)
            # Trebuie să mapăm corect indexul 0-3 la structura internă
            v_lane_idx = 0
            try:
                if v.lane_index[0] == "a":
                    v_lane_idx = v.lane_index[2]
                else:
                    # Benzile 0,1 de pe "b"-> "a" corespund vizual cu 2,3
                    v_lane_idx = v.lane_index[2] + 2
            except:
                continue
                
            if v_lane_idx != lane_idx_check:
                continue

            # Calculăm distanța pe axa X (longitudinală)
            d_raw = v.position[0] - ego_pos[0]
            
            # Determinăm limita de căutare în spate
            # Pentru banda curentă: ne interesează doar ce e în față (sau foarte puțin suprapus)
            # Pentru alte benzi: ne interesează și ce e în "unghiul mort" sau lateral (-20m)
            search_behind_dist = -10.0 if lane_idx_check != ego_lane else -2.0
            
            if d_raw > search_behind_dist and d_raw < closest_dist:
                closest_dist = d_raw
                closest_v = v

    # Dacă nu e nimeni în față pe distanța relevantă -> SAFE
    if closest_v is None:
        return SIT_SAFE

    # Calculăm situația bazat pe distanță și viteză relativă
    if is_oncoming_lane:
        # Pe contrasens, viteza relativă e suma vitezelor (aproximativ)
        # E mult mai periculos
        if closest_dist < DIST_DANGER_ONCOMING:
            return SIT_DANGER
        elif closest_dist < DIST_CAUTION_ONCOMING:
            return SIT_CAUTION
        else:
            return SIT_SAFE
    else:
        # Sens normal
        rel_speed = ego_speed - closest_v.speed # Pozitiv = ne apropiem
        
        if closest_dist < DIST_DANGER_SAME:
            return SIT_DANGER
        elif closest_dist < DIST_CAUTION_SAME:
            # Dacă ne apropiem repede, e Danger chiar dacă distanța e medie
            if rel_speed > 5.0: # Ne apropiem cu > 18 km/h diferență
                return SIT_DANGER
            return SIT_CAUTION
        else:
            # Distanță mare, dar verificăm viteza relativă extremă
            if rel_speed > 10.0 and closest_dist < 80.0:
                return SIT_CAUTION
            return SIT_SAFE


def discretize_state(env) -> Tuple[int, int, int, int, int, int]:
    """
    Mapează starea la o tuplă discretă:
    (lane_idx, speed_bin, sit_l0, sit_l1, sit_l2, sit_l3)
    """
    # 1. Lane index (0-3)
    lane_idx = 0
    if hasattr(env, "vehicle") and getattr(env.vehicle, "lane_index", None):
        try:
            lane_idx = int(env.vehicle.lane_index[2])
        except Exception:
            lane_idx = 0
    lane_idx = max(0, min(int(ENV_CONFIG.get("lanes_count", 4)) - 1, lane_idx))

    # 2. Speed bin (0-2)
    speed = float(getattr(env.vehicle, "speed", 0.0))
    speed_bin = int(math.floor((speed - 0.01) / SPEED_BIN))
    speed_bin = max(0, min(MAX_SPEED_BINS - 1, speed_bin))

    # 3. Situations per lane
    ego_pos = env.vehicle.position
    
    sit_l0 = get_lane_situation(env, 0, ego_pos, speed, lane_idx)
    sit_l1 = get_lane_situation(env, 1, ego_pos, speed, lane_idx)
    sit_l2 = get_lane_situation(env, 2, ego_pos, speed, lane_idx)
    sit_l3 = get_lane_situation(env, 3, ego_pos, speed, lane_idx)

    return (lane_idx, speed_bin, sit_l0, sit_l1, sit_l2, sit_l3)


def epsilon_greedy(Q: np.ndarray, state: Tuple[int, int, int, int, int, int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(Q.shape[-1])
    lane, spd, s0, s1, s2, s3 = state
    return int(np.argmax(Q[lane, spd, s0, s1, s2, s3]))


def train_q_learning(episodes: int = 200, max_steps: int = 2000, alpha: float = 0.2, gamma: float = 0.99,
                     eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.997,
                     Q_init: np.ndarray = None):
    """Antrenare Q-learning îmbunătățit."""
    env = make_env(render_mode=None)
    n_actions = env.action_space.n
    lanes = int(ENV_CONFIG.get("lanes_count", 4))
    
    # Folosește Q-table existent sau creează unul nou
    if Q_init is not None:
        Q = Q_init
        print(f"[INFO] Q-Table încărcat: {Q.shape}")
    else:
        # OPTIMISTIC INITIALIZATION
        # Shape: (4, 3, 3, 3, 3, 3, 5)
        # (lane, speed, sit0, sit1, sit2, sit3, action)
        Q = np.ones((lanes, MAX_SPEED_BINS, NUM_SITUATIONS, NUM_SITUATIONS, NUM_SITUATIONS, NUM_SITUATIONS, n_actions), dtype=np.float32) * 5.0
        # Acțiunile de lane change primesc bonus inițial
        Q[:, :, :, :, :, :, 0] = 8.0  # LANE_LEFT
        Q[:, :, :, :, :, :, 2] = 8.0  # LANE_RIGHT

    rewards = []
    lane_changes_per_ep = []  # Track lane changes
    epsilon = eps_start
    print(f"Start Training... Q-Table size: {Q.size} elemente.")
    print(f"[CONFIG] alpha={alpha}, gamma={gamma}, eps_decay={eps_decay}")
    print(f"[CONFIG] Optimistic init: Q-values start at 5.0-8.0")

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            state = discretize_state(env)
            total_reward = 0.0
            lane_changes = 0  # Counter pentru acest episod
            prev_lane = state[0]

            for t in range(max_steps):
                action = epsilon_greedy(Q, state, epsilon)
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = discretize_state(env)
                
                # Track lane changes
                if next_state[0] != prev_lane:
                    lane_changes += 1
                    prev_lane = next_state[0]

                l, s, s0, s1, s2, s3 = state
                nl, ns, ns0, ns1, ns2, ns3 = next_state
                
                best_next = np.max(Q[nl, ns, ns0, ns1, ns2, ns3])
                td_target = reward + gamma * best_next * (0 if terminated or truncated else 1)
                td_error = td_target - Q[l, s, s0, s1, s2, s3, action]
                Q[l, s, s0, s1, s2, s3, action] += alpha * td_error

                total_reward += reward
                state = next_state
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            lane_changes_per_ep.append(lane_changes)
            epsilon = max(eps_end, epsilon * eps_decay)
            print(f"Ep {ep + 1:04d} | eps={epsilon:.3f} | steps={t+1} | LC={lane_changes:2d} | reward={total_reward:.2f}")
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
    parser.add_argument("--max-steps", type=int, default=2000, help="Pași pe episod")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Epsilon inițial")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Epsilon minim")
    parser.add_argument("--eps-decay", type=float, default=0.992, help="Factor de decay")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episoade de evaluare")
    parser.add_argument("--eval-render", action="store_true", help="Randează în evaluare")
    parser.add_argument("--model-path", type=str, default="q_table.npy", help="Path model salvat")
    parser.add_argument("--load-model", type=str, default=None, help="Path model de încărcat")
    parser.add_argument("--eval-only", action="store_true", help="Doar evaluare")
    args = parser.parse_args()

    if args.eval_only:
        try: Q = load_model(args.model_path)
        except FileNotFoundError: return
    else:
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