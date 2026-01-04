import gymnasium as gym
from gymnasium.envs.registration import register
from highway_env import utils
from highway_env.envs.two_way_env import TwoWayEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork

from env_config import ENV_CONFIG


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


# Înregistrare pentru gym.make
register(
    id="two-way-4lane-v0",
    entry_point=TwoWay4LaneEnv,
)


def make_env(render_mode: str = "human") -> gym.Env:
    """Helper pentru a crea env-ul cu config pre-set."""
    return gym.make("two-way-4lane-v0", render_mode=render_mode, config=ENV_CONFIG)


__all__ = ["TwoWay4LaneEnv", "make_env"]
