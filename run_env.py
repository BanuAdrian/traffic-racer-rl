"""Rulează random rollout pentru env-ul custom separat de definiția env/config."""

import argparse
import time

from traffic_racer_env import make_env


def rollout(env, steps: int = 800) -> None:
    obs, info = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if env.render_mode == "human":
            env.render()
            time.sleep(0.2)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Rulează un episod random pe two-way-4lane-v0")
    parser.add_argument("--steps", type=int, default=800, help="Număr de pași în rollout")
    parser.add_argument("--no-render", action="store_true", help="Rulează fără randare")
    args = parser.parse_args()

    env = make_env(render_mode=None if args.no_render else "human")
    env.metadata["render_fps"] = 30
    rollout(env, steps=args.steps)
if __name__ == "__main__":
    main()
