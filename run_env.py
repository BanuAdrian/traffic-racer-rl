"""Rulează random rollout pentru env-ul custom separat de definiția env/config."""

import argparse
import time

from traffic_racer_env import make_env


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
        
        # Luăm config-ul ca să știm cât valorează fiecare acțiune
        config = env.unwrapped.config

        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) 
            total_reward += reward
            
            if hasattr(env.unwrapped, "vehicle") and env.unwrapped.vehicle:
                ego = env.unwrapped.vehicle
                
                # --- 1. Distanță ---
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
                dist_str = f"{min_dist:.1f}m" if min_dist != float("inf") else "Free"

                # --- 2. Reward-uri Reale (Cantitate * Preț) ---
                r_dict = env.unwrapped._rewards(1) 
                
                # Aici facem conversia: Cantitate (din dict) * Preț (din config)
                hero_pts = r_dict.get('oncoming_overtake_reward', 0.0) * config.get("oncoming_overtake_reward", 0.0)
                near_pts = r_dict.get('near_miss_reward', 0.0) * config.get("near_miss_reward", 0.0)
                ovr_pts  = r_dict.get('overtaking_reward', 0.0) * config.get("overtaking_reward", 0.0)
                path_pts = r_dict.get('clear_path_reward', 0.0) * config.get("clear_path_reward", 0.0)
                
                # La viteză calculul e mai complex, afișăm doar procentul (0.0 la 1.0)
                spd_val  = r_dict.get('high_speed_reward', 0.0)

                # Info bandă
                try:
                    lane_idx = ego.lane_index[2]
                    lane_type = "ONC" if lane_idx >= 2 else "OK "
                except:
                    lane_idx = -1; lane_type = "???"

                debug_str = (
                    f"Score:{total_reward:.2f}|"
                    f"HERO:{hero_pts:.0f}|"    # Ar trebui să afișeze 4 sau 0
                    f"Near:{near_pts:.1f}|"    # Ar trebui să afișeze 0.5, 1.0 etc
                    f"Ovr:{ovr_pts:.0f}|"      # 1 sau 0
                    f"Path:{path_pts:.1f}|"
                    f"Spd:{spd_val:.2f}|"
                    f"L:{lane_idx}({lane_type})|"
                    f"Dist:{dist_str:<5}"
                )
                print(debug_str + " " * 5, end="\r")

            env.render()
            if terminated or truncated:
                print(f"\n[Terminated] Final Score: {total_reward:.2f}. Resetare...")
                env.reset()
                total_reward = 0.0
    else:
        print(f"Env config manual_control: {env.unwrapped.config.get('manual_control')}")
        env.metadata["render_fps"] = 30
        rollout(env, steps=args.steps)
if __name__ == "__main__":
    main()
