import argparse
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import time
from highway_env import utils
from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork

# Config aliniat cu ppo.py (lanes_count este pe direcție în TwoWayEnv)
ENV_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 25,
    "duration": 40,
    "controlled_vehicles": 1,
    "ego_spacing": 2,
    "collision_reward": -1.0,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "offroad_terminal": True,
    "road_length": 800,
    "screen_width": 1400,
    "screen_height": 800,
    "scaling": 30,
    "centering_position": [0.35, 0.55],
}


class TwoWay4LaneEnv(TwoWayEnv):
    """Two-way cu 2 benzi pe sens, compatibil cu setup-ul din ppo.py."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(ENV_CONFIG)
        return config

    def _make_road(self, length: int | None = None):  # type: ignore[override]
        length = length or self.config.get("road_length", 800)
        w = StraightLane.DEFAULT_WIDTH
        net = RoadNetwork()

        # Forward lanes (a -> b)
        net.add_lane(
            "a", "b",
            StraightLane([0, 0], [length, 0], line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)),
        )
        net.add_lane(
            "a", "b",
            StraightLane([0, w], [length, w], line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)),
        )
        net.add_lane(
            "a", "b",
            StraightLane([0, 2 * w], [length, 2 * w], line_types=(LineType.NONE, LineType.STRIPED)),
        )
        net.add_lane(
            "a", "b",
            StraightLane([0, 3 * w], [length, 3 * w], line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)),
        )

        # Oncoming lanes (b -> a) suprapuse spațial cu ultimele două forward lanes
        net.add_lane(
            "b", "a",
            StraightLane([length, 2 * w], [0, 2 * w], line_types=(LineType.NONE, LineType.NONE)),
        )
        net.add_lane(
            "b", "a",
            StraightLane([length, 3 * w], [0, 3 * w], line_types=(LineType.NONE, LineType.NONE)),
        )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),
        )

    def _make_vehicles(self) -> None:  # type: ignore[override]
        road = self.road

        # Ego pe lane 1 (a->b, banda rapidă forward)
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
        )
        ego_vehicle.enable_lane_change = True
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Trafic în sensul nostru (a->b lane 0 și 1)
        for i in range(6):
            lane_idx = int(self.np_random.integers(0, 2))
            lane = ("a", "b", lane_idx)
            road.vehicles.append(
                vehicles_type(
                    road,
                    position=road.network.get_lane(lane).position(
                        60.0 + 60.0 * float(i) + 5.0 * self.np_random.normal(), 0.0
                    ),
                    heading=road.network.get_lane(lane).heading_at(0.0),
                    speed=20.0 + 2.0 * self.np_random.normal(),
                    enable_lane_change=False,  # NPC-urile albastre rămân pe sensul lor
                )
            )

        # Trafic din sens opus (b->a lane 0 și 1)
        for i in range(6):
            lane_idx = int(self.np_random.integers(0, 2))
            lane = ("b", "a", lane_idx)
            v = vehicles_type(
                road,
                position=road.network.get_lane(lane).position(
                    200.0 + 90.0 * float(i) + 10.0 * self.np_random.normal(), 0
                ),
                heading=road.network.get_lane(lane).heading_at(0.0),
                speed=22.0 + 3.0 * self.np_random.normal(),
                enable_lane_change=False,
            )
            v.target_lane_index = lane
            road.vehicles.append(v)

    def _is_truncated(self) -> bool:  # type: ignore[override]
        # Episodul se termină când ego ajunge aproape de capătul benzii.
        lane = self.vehicle.lane
        s, _ = lane.local_coordinates(self.vehicle.position)
        return s >= lane.length - 5.0


# Înregistrare opțională pentru gym.make
register(
    id="two-way-4lane-local-v0",
    entry_point=TwoWay4LaneEnv,
)


def make_env(render_mode: str = "human") -> gym.Env:
    """Helper ca în ppo.py pentru a crea env-ul cu config pre-set."""
    return gym.make("two-way-4lane-local-v0", render_mode=render_mode, config=ENV_CONFIG)


def rollout(env: gym.Env, steps: int = 500) -> None:
    obs, info = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if env.render_mode == "human":
            env.render()
            time.sleep(0.2)  # slow down vizualizarea
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", help="Rulează fără randare")
    args = parser.parse_args()

    env = make_env(render_mode=None if args.no_render else "human")
    env.metadata["render_fps"] = 30
    rollout(env, steps=800)
