"""Rulează random rollout pentru env-ul custom separat de definiția env/config."""

import argparse
import time

from traffic_racer_env import make_env


def rollout(env, steps: int = 800) -> None:
    obs, info = env.reset()
    print("Rollout start. Action space:", env.action_space)
    for i in range(steps):
        action = env.action_space.sample()
        # print(f"Step {i}, Action: {action}") # Uncomment for debug
        obs, reward, terminated, truncated, info = env.step(action)
        if env.render_mode == "human":
            env.render()
            time.sleep(0.01)
        if terminated or truncated:
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
        # Reconfigurează pentru control manual
        env.unwrapped.configure({"manual_control": True})
        env.reset()
        print("Control manual activat. Folosește săgețile pentru a conduce.")
        
        done = False
        while not done:
            # În mod manual, step() preia acțiunea din pygame events dacă manual_control=True
            # Dar highway-env are nevoie de o acțiune dummy sau None
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) 
            env.render()
            # Nu mai facem sleep aici, pygame se ocupă de timing
            if terminated or truncated:
                env.reset()
    else:
        print(f"Env config manual_control: {env.unwrapped.config.get('manual_control')}")
        env.metadata["render_fps"] = 30
        rollout(env, steps=args.steps)
if __name__ == "__main__":
    main()
