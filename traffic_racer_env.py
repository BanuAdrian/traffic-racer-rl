import gymnasium as gym
import numpy as np
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
        length = self.config.get("road_length", 2000)
        vehicles_count = self.config.get("vehicles_count", 50)
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Ego pe lane 1 (a->b, banda rapidă forward)
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
        )
        # Asigură-te că ego nu e controlat manual dacă config-ul zice nu
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        if vehicles_count <= 0:
            return

        # Generăm toate sloturile posibile pentru a evita coliziunile la spawn
        # Distanța minimă între mașini (buffer)
        min_spacing = 25 
        
        potential_slots = []
        
        # Sensul nostru (a->b) - start de la 60m ca să nu fie peste ego
        for x in range(60, length, min_spacing):
            for lane_idx in range(2):
                potential_slots.append(("forward", lane_idx, x))
                
        # Sens opus (b->a)
        for x in range(0, length, min_spacing):
            for lane_idx in range(2):
                potential_slots.append(("oncoming", lane_idx, x))
                
        # Amestecăm sloturile și alegem primele vehicles_count
        # Folosim np_random pentru reproductibilitate
        # Convertim la listă de indici pentru a putea face shuffle
        indices = np.arange(len(potential_slots))
        self.np_random.shuffle(indices)
        
        # Luăm doar câte avem nevoie
        indices = indices[:vehicles_count]
        
        for i in indices:
            direction, lane_idx, x_pos = potential_slots[i]
            
            # Adăugăm puțin zgomot la poziție
            noise = self.np_random.uniform(-5, 5)
            pos = x_pos + noise
            if pos < 0 or pos >= length:
                continue
                
            if direction == "forward":
                lane = ("a", "b", lane_idx)
                speed = 20.0 + 5.0 * self.np_random.normal()
            else:
                lane = ("b", "a", lane_idx)
                speed = 22.0 + 5.0 * self.np_random.normal()
                
            v = vehicles_type(
                road,
                position=road.network.get_lane(lane).position(pos, 0.0),
                heading=road.network.get_lane(lane).heading_at(pos),
                speed=speed,
                enable_lane_change=False,
            )
            road.vehicles.append(v)

    def _rewards(self, action: int) -> dict:
        """Calculăm componentele reward-ului."""
        # Viteza normalizată (0 la 1)
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "oncoming_lane_reward": float(self.vehicle.lane_index[2] >= 2),
        }

    def _reward(self, action: int) -> float:
        """Agregăm reward-urile ponderate."""
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * value 
            for name, value in rewards.items()
        )
        # Penalizare coliziune directă (dacă nu e inclusă în sumă corect sau vrem să fim siguri)
        # De obicei collision_reward e negativ în config, deci sum() e ok.
        return reward

    def _is_truncated(self) -> bool:  # type: ignore[override]
        # Episodul se termină când ego ajunge aproape de capătul benzii.
        lane = self.vehicle.lane
        s, _ = lane.local_coordinates(self.vehicle.position)
        return s >= lane.length - 5.0


# Înregistrare pentru gym.make
register(
    id="two-way-4lane-v1",
    entry_point=TwoWay4LaneEnv,
)


def make_env(render_mode: str = "human") -> gym.Env:
    """Helper pentru a crea env-ul cu config pre-set."""
    config = ENV_CONFIG.copy()
    config["manual_control"] = False  # Force disable manual control
    return gym.make("two-way-4lane-v1", render_mode=render_mode, config=config)


__all__ = ["TwoWay4LaneEnv", "make_env"]
