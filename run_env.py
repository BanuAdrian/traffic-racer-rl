"""Rulează random rollout pentru env-ul custom separat de definiția env/config."""

import argparse
import time
import math
from collections import deque

from traffic_racer_env import make_env
from q_learning import discretize_state

# Buffer pentru smoothing distanță (medie mobilă pe ultimele N valori)
DIST_BUFFER_SIZE = 10
dist_buffer = deque(maxlen=DIST_BUFFER_SIZE)


def rollout(env, steps: int = 800) -> None:
    obs, info = env.reset()
    print("Rollout start. Action space:", env.action_space)
    for i in range(steps):
        action = env.action_space.sample()
        print(f"Step {i}, Action: {action}") # Uncomment for debug
        obs, reward, terminated, truncated, info = env.step(action)
        if env.render_mode == "human":
            env.render()
            time.sleep(0.01)
        if terminated or truncated:
            print("terminated or truncated")
            obs, info = env.reset()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Rulează un episod random pe two-way-4lane-v0")
    parser.add_argument("--steps", type=int, default=800, help="Număr de pași în rollout")
    parser.add_argument("--no-render", action="store_true", help="Rulează fără randare")
    parser.add_argument("--manual", action="store_true", help="Activează control manual")
    args = parser.parse_args()

    env = make_env(render_mode=None if args.no_render else "human")
    
    if args.manual:
        env.unwrapped.configure({"manual_control": True})
        env.reset()
        print("Control manual activat. Folosește săgețile.")
        
        total_reward = 0.0
        done = False
        step_count = 0  # Counter pentru step-uri
        
        # Luăm config-ul ca să știm cât valoreaază fiecare acțiune
        config = env.unwrapped.config

        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) 
            total_reward += reward
            step_count += 1
            
            if hasattr(env.unwrapped, "vehicle") and env.unwrapped.vehicle:
                ego = env.unwrapped.vehicle
                
                # --- 1. Distanță cu SMOOTHING ---
                min_dist = float("inf")
                if hasattr(env.unwrapped, "road") and env.unwrapped.road:
                    for v in env.unwrapped.road.vehicles:
                        if v is ego: continue
                        try:
                            if v.lane_index[2] == ego.lane_index[2]:
                                d_long = v.position[0] - ego.position[0]
                                if 0 < d_long < min_dist:
                                    min_dist = d_long
                        except: pass
                
                # Adaugă în buffer și calculează media
                if min_dist != float("inf"):
                    dist_buffer.append(min_dist)
                    avg_dist = sum(dist_buffer) / len(dist_buffer)
                    dist_str = f"{avg_dist:.0f}m"  # Afișăm fără zecimale pentru stabilitate
                else:
                    dist_buffer.clear()  # Reset buffer când e liber
                    dist_str = "Free"

                # --- 2. Reward-uri Reale (Cantitate * Preț) ---
                r_dict = env.unwrapped._rewards(1) 
                
                # Calculăm toate rewardurile din step-ul curent
                hero_pts = r_dict.get('oncoming_overtake_reward', 0.0) * config.get("oncoming_overtake_reward", 0.0)
                near_pts = r_dict.get('near_miss_reward', 0.0) * config.get("near_miss_reward", 0.0)
                ovr_pts  = r_dict.get('overtaking_reward', 0.0) * config.get("overtaking_reward", 0.0)
                path_pts = r_dict.get('clear_path_reward', 0.0) * config.get("clear_path_reward", 0.0)
                spd_pts  = r_dict.get('high_speed_reward', 0.0) * config.get("high_speed_reward", 0.0)
                onc_pts  = r_dict.get('oncoming_lane_reward', 0.0) * config.get("oncoming_lane_reward", 0.0)
                stag_pts = r_dict.get('stagnation_penalty', 0.0) * config.get("stagnation_penalty", 0.0)
                prog_pts = r_dict.get('progress_reward', 0.0) * config.get("progress_reward", 0.0)
                fin_pts  = r_dict.get('finish_reward', 0.0) * config.get("finish_reward", 0.0)
                lane_pts = r_dict.get('lane_change_reward', 0.0) * config.get("lane_change_reward", 0.0)

                # Info bandă
                try:
                    lane_idx = ego.lane_index[2]
                    lane_type = "ONC" if lane_idx >= 2 else "OK "
                except:
                    lane_idx = -1; lane_type = "???"
                
                # Progres pe drum
                road_len = config.get("road_length", 1250)
                progress_pct = (ego.position[0] / road_len) * 100

                # Linia 1: Info generală
                line1 = (
                    f"Step:{step_count:4d} | "
                    f"Score:{total_reward:7.1f} | "
                    f"Spd:{ego.speed:4.1f} | "
                    f"L:{lane_idx}({lane_type}) | "
                    f"Dist:{dist_str:<5} | "
                    f"Prog:{progress_pct:5.1f}%"
                )
                
                # Linia 2: Rewarduri din step-ul curent (doar cele nenule)
                rewards_parts = []
                if hero_pts != 0: rewards_parts.append(f"HERO:{hero_pts:+.1f}")
                if near_pts != 0: rewards_parts.append(f"Near:{near_pts:+.1f}")
                if ovr_pts != 0: rewards_parts.append(f"Ovr:{ovr_pts:+.1f}")
                if path_pts != 0: rewards_parts.append(f"Path:{path_pts:+.2f}")
                if spd_pts != 0: rewards_parts.append(f"Spd:{spd_pts:+.2f}")
                if onc_pts != 0: rewards_parts.append(f"Onc:{onc_pts:+.2f}")
                if stag_pts != 0: rewards_parts.append(f"STAG:{stag_pts:+.1f}")
                if prog_pts != 0: rewards_parts.append(f"Prog:{prog_pts:+.2f}")
                if lane_pts != 0: rewards_parts.append(f"Lane:{lane_pts:+.2f}")
                if fin_pts != 0: rewards_parts.append(f"FIN:{fin_pts:+.1f}")
                
                line2 = "Rewards: " + " | ".join(rewards_parts) if rewards_parts else "Rewards: (none)"
                
                # Afișăm pe două linii
                disc = discretize_state(env.unwrapped)
                # Afișează totul pe o singură linie, cu padding și \r
                print(f"\r{line1} | Disc:{disc}    ", end="")
                print(f"\n{line2}" + " " * 40, end="")
                print("\033[F", end="")  # Mută cursorul însus cu o linie

            env.render()
            if terminated or truncated:
                print(f"\n\n[Terminated] Step: {step_count} | Final Score: {total_reward:.2f}. Resetare...")
                env.reset()
                total_reward = 0.0
                step_count = 0
                dist_buffer.clear()
    else:
        print(f"Env config manual_control: {env.unwrapped.config.get('manual_control')}")
        env.metadata["render_fps"] = 30
        rollout(env, steps=args.steps)
if __name__ == "__main__":
    main()
