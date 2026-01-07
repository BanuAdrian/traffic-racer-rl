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
        length = self.config.get("road_length", 1500)
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        base_speed = self.config.get("other_vehicles_speed", 20.0)

        # 1. Creăm EGO (Agentul)
        ego_lane_idx = 1
        ego_vehicle = self.action_type.vehicle_class(
            road, 
            road.network.get_lane(("a", "b", ego_lane_idx)).position(30.0, 0.0), 
            speed=self.config.get("ego_init_speed", 20.0)
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # 2. Generăm trafic inițial RANDOMIZAT
        # Număr de mașini proporțional cu lungimea drumului
        min_initial = self.config.get("min_initial_vehicles", 8)
        max_initial = self.config.get("max_initial_vehicles", 20)
        
        # Scalăm numărul de mașini cu lungimea drumului (bază = 1500m)
        length_multiplier = length / 1500.0
        scaled_min = int(min_initial * length_multiplier)
        scaled_max = int(max_initial * length_multiplier)
        
        initial_vehicles = self.np_random.integers(scaled_min, scaled_max + 1)
        
        # Creăm liste cu poziții ocupate per bandă pentru verificare rapidă
        self._lane_positions = {
            ("a", "b", 0): [],
            ("a", "b", 1): [],
            ("b", "a", 0): [],
            ("b", "a", 1): [],
        }
        
        # Spawnăm mașini pe SENSUL NOSTRU (albastre) - distribuite random
        same_direction_count = int(initial_vehicles * 0.90)  # ~90% pe sensul nostru, doar 10% contrasens
        ego_x = self.vehicle.position[0]  # ~30
        
        for _ in range(same_direction_count):
            # Poziție randomă în fața agentului (X global > ego_x + 50)
            x_pos = self.np_random.uniform(ego_x + 50, min(length - 100, ego_x + 800))
            lane_id = ("a", "b", self.np_random.integers(0, 2))
            speed = self.np_random.uniform(base_speed * 0.85, base_speed * 1.1)
            # Spacing VARIABIL: între 15m și 35m pentru aspect natural
            var_spacing = self.np_random.uniform(15.0, 35.0)
            self._spawn_vehicle_safe(lane_id, x_pos, speed, vehicles_type, min_spacing=var_spacing)
        
        # Spawnăm mașini pe CONTRASENS (galbene) - pe TOT drumul în fața agentului!
        # IMPORTANT: Pentru contrasens, coordonata locală 's' merge INVERS:
        # s=0 -> X global = length (capătul drept)
        # s=length -> X global = 0 (capătul stâng)
        # Deci pentru X global = ego_x + 50, s = length - (ego_x + 50)
        oncoming_count = initial_vehicles - same_direction_count
        
        for _ in range(oncoming_count):
            # Vrem X global între ego_x + 60 și length - 50
            # Deci s între length - (length - 50) = 50 și length - (ego_x + 60)
            min_s = 50  # X global = length - 50
            max_s = length - (ego_x + 60)  # X global = ego_x + 60
            
            if min_s < max_s:
                s_pos = self.np_random.uniform(min_s, max_s)
                lane_id = ("b", "a", self.np_random.integers(0, 2))
                speed = self.np_random.uniform(base_speed * 0.9, base_speed * 1.1)
                # Spacing VARIABIL pentru contrasens: 30-60m
                var_spacing = self.np_random.uniform(30.0, 60.0)
                self._spawn_vehicle_safe(lane_id, s_pos, speed, vehicles_type, min_spacing=var_spacing)

        self.last_step_overtaken = 0
        self.last_step_oncoming_passed = 0
        self._spawn_cooldown = 0
        # Counter pentru cât timp stă pe aceeași bandă
        self._same_lane_steps = 0
        self._last_lane = None
        
    def _spawn_vehicle_safe(self, lane_id: tuple, x_pos: float, speed: float, 
                            vehicles_type, min_spacing: float = 25.0) -> bool:
        """Spawnează o mașină DOAR dacă e sigur. Verifică toate coliziunile posibile."""
        road = self.road
        
        # Obținem poziția reală în coordonate globale ÎNAINTE de orice verificare
        lane_obj = road.network.get_lane(lane_id)
        proposed_pos = lane_obj.position(x_pos, 0.0)
        real_x = proposed_pos[0]  # Poziția X reală în lume
        
        # Verificare 1: Nu spawna prea aproape de alte mașini de pe ACEEAȘI BANDĂ
        for existing_x in self._lane_positions.get(lane_id, []):
            if abs(existing_x - x_pos) < min_spacing:
                return False
        
        # Verificare 2: Pentru mașini din contrasens, verifică și banda adiacentă
        if lane_id[0] == "b":  # Contrasens
            adjacent_lane = ("b", "a", 1 - lane_id[2])
            for existing_x in self._lane_positions.get(adjacent_lane, []):
                if abs(existing_x - x_pos) < min_spacing * 0.7:
                    return False
        
        # Verificare 3: Nu spawna în spatele sau prea aproape de agent
        ego_x = self.vehicle.position[0]
        
        # Pentru ORICE bandă: mașina trebuie să fie în FAȚA agentului
        if real_x < ego_x + 40:  # Minim 40m în față
            return False
        
        # Verificare 4: Double-check cu toate mașinile existente
        for v in road.vehicles:
            if v is self.vehicle:
                continue
            dx = abs(v.position[0] - real_x)
            dy = abs(v.position[1] - proposed_pos[1])
            if dy < 3.0 and dx < min_spacing:
                return False
        
        try:
            heading = lane_obj.heading_at(x_pos)
            v = vehicles_type(
                road,
                position=proposed_pos,
                heading=heading,
                speed=speed,
                enable_lane_change=False,
            )
            road.vehicles.append(v)
            self._lane_positions[lane_id].append(x_pos)
            return True
        except Exception:
            return False
    
    def _dynamic_traffic_spawn(self):
        """Generează trafic dinamic în fața agentului pentru flux continuu."""
        if self._spawn_cooldown > 0:
            self._spawn_cooldown -= 1
            return
            
        road = self.road
        length = self.config.get("road_length", 1500)
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        base_speed = self.config.get("other_vehicles_speed", 20.0)
        
        ego_x = self.vehicle.position[0]
        
        # Densitate țintă în zona vizibilă
        target_density = self.config.get("traffic_density", 10)
        visible_range = 150.0
        
        # Numără mașinile din zona vizibilă în FAȚĂ
        forward_vehicles = [
            v for v in road.vehicles 
            if v is not self.vehicle 
            and v.lane_index[0] == "a"  # Sensul nostru
            and ego_x < v.position[0] < ego_x + visible_range
        ]
        
        oncoming_vehicles = [
            v for v in road.vehicles 
            if v is not self.vehicle 
            and v.lane_index[0] == "b"  # Contrasens
            and ego_x < v.position[0] < ego_x + visible_range * 1.5
        ]
        
        # Spawn mașini pe sensul nostru (doar dacă nu suntem prea aproape de final)
        if ego_x < length - 200 and len(forward_vehicles) < target_density * 0.7:
            spawn_x = ego_x + self.np_random.uniform(110, 180)
            if spawn_x < length - 50:
                lane_id = ("a", "b", self.np_random.integers(0, 2))
                speed = self.np_random.uniform(base_speed * 0.8, base_speed * 1.05)
                # Spacing variabil: 18-40m
                var_spacing = self.np_random.uniform(18.0, 40.0)
                self._spawn_vehicle_safe(lane_id, spawn_x, speed, vehicles_type, min_spacing=var_spacing)
        
        # Spawn mașini din contrasens - FOARTE PUȚINE (10% din densitate)
        # Astfel ai timp să depășești pe contrasens între ele
        # IMPORTANT: Pentru contrasens trebuie să convertim X global în coordonată locală 's'
        # s = length - X_global
        if len(oncoming_vehicles) < target_density * 0.3:
            # Spawnăm la distanță mare pentru a lăsa timp de reacție
            target_x_min = ego_x + 150
            target_x_max = min(ego_x + 250, length - 10)
            
            if target_x_min < target_x_max:
                target_x = self.np_random.uniform(target_x_min, target_x_max)
                # Convertim în coordonată locală pentru contrasens
                s_pos = length - target_x
                lane_id = ("b", "a", self.np_random.integers(0, 2))
                speed = self.np_random.uniform(base_speed * 0.9, base_speed * 1.1)
                # Spacing variabil pentru contrasens: 40-70m
                var_spacing = self.np_random.uniform(40.0, 70.0)
                self._spawn_vehicle_safe(lane_id, s_pos, speed, vehicles_type, min_spacing=var_spacing)
        
        self._spawn_cooldown = 5  # Check mai rar (la fiecare 5 frame-uri)

    def step(self, action: int):
        # 0. SPAWN DINAMIC - Generează trafic nou
        self._dynamic_traffic_spawn()
        
        # 1. Identificăm cine e în față ÎNAINTE de mișcare
        
        # A. Cei pe sensul nostru (pentru depășire normală)
        vehicles_in_front = [
            v for v in self.road.vehicles 
            if v is not self.vehicle 
            and v.position[0] > self.vehicle.position[0]
            and v.lane_index[0] == "a"  # Sensul a->b
        ]
        
        # B. Cei de pe CONTRASENS (pentru Near Miss)
        oncoming_in_front = [
            v for v in self.road.vehicles
            if v is not self.vehicle
            and v.position[0] > self.vehicle.position[0]
            and v.lane_index[0] == "b"  # Sensul b->a (VIN DIN FAȚĂ)
        ]
        
        # 2. Executăm pasul fizic
        result = super().step(action)
        
        # 3. --- CURĂȚENIE GENERALĂ (GARBAGE COLLECTOR) ---
        road_len = self.config.get("road_length", 1500)
        ego_x = self.vehicle.position[0]
        
        # Resetăm pozițiile pentru următorul frame
        self._lane_positions = {
            ("a", "b", 0): [],
            ("a", "b", 1): [],
            ("b", "a", 0): [],
            ("b", "a", 1): [],
        }
        
        # Iterăm printr-o COPIE a listei
        for v in self.road.vehicles[:]:
            if v is not self.vehicle:
                # Ștergem mașinile care au ieșit din drum sau sunt prea în urmă
                should_remove = False
                
                # Mașini care au ajuns la capătul drumului
                if v.position[0] >= road_len - 10.0:
                    should_remove = True
                # Mașini din contrasens care au trecut de noi (sunt în spatele nostru)
                elif v.lane_index[0] == "b" and v.position[0] < ego_x - 50:
                    should_remove = True
                # Mașini din sensul nostru care sunt prea în urmă
                elif v.lane_index[0] == "a" and v.position[0] < ego_x - 100:
                    should_remove = True
                    
                if should_remove:
                    self.road.vehicles.remove(v)
                else:
                    # Actualizăm pozițiile pentru spawn check
                    lane_id = v.lane_index
                    if lane_id in self._lane_positions:
                        self._lane_positions[lane_id].append(v.position[0])
                    
        # 4. Verificăm cine a ajuns în spate DUPĂ mișcare
        
        # Calcul depășiri normale
        overtaken = 0
        for v in vehicles_in_front:
            if v.position[0] < self.vehicle.position[0]:
                overtaken += 1
        self.last_step_overtaken = overtaken
        
        # Calcul treceri pe lângă contrasens (Near Miss)
        oncoming_passed = 0
        for v in oncoming_in_front:
            if v.position[0] < self.vehicle.position[0]:
                oncoming_passed += 1
        self.last_step_oncoming_passed = oncoming_passed
        
        return result

    def _rewards(self, action: int) -> dict:
        """
        Sistem de rewarduri pentru Traffic Racer:
        - Ajunge la final
        - Face multe depășiri  
        - Menține highest speed
        - Merge pe contrasens des
        - NU stă în spatele mașinilor lente
        - NU stă pe aceeași bandă prea mult
        """
        
        current_lane_idx = 0
        try: current_lane_idx = self.vehicle.lane_index[2]
        except: pass
        
        # --- TRACK SAME LANE TIME ---
        if self._last_lane is None:
            self._last_lane = current_lane_idx
        
        if current_lane_idx == self._last_lane:
            self._same_lane_steps += 1
        else:
            self._same_lane_steps = 0
            self._last_lane = current_lane_idx
        
        # Penalizare crescătoare pentru a sta pe aceeași bandă prea mult
        # După 50 steps (~10 secunde) începe penalizarea
        same_lane_penalty = 0.0
        if self._same_lane_steps > 50:
            # Penalizare care crește liniar, ajungând la maxim în ~10 secunde (50 steps)
            same_lane_penalty = min(1.0, (self._same_lane_steps - 50) / 50.0)
        
        # Ești pe contrasens dacă indexul e 2 sau 3
        is_oncoming_lane = 1.0 if current_lane_idx >= 2 else 0.0

        # --- 1. VITEZA ---
        # Folosim viteza maximă din config pentru normalizare
        target_speeds = self.config.get("action", {}).get("target_speeds", [5, 10, 15])
        max_speed = max(target_speeds) if target_speeds else 15.0
        current_speed = self.vehicle.speed
        
        # Reward pătratic pentru viteză (încurajează viteza maximă)
        speed_reward = (current_speed / max_speed) ** 2
        
        # --- 2. PENALIZARE STAGNARE ---
        # Dacă ai mașină în față aproape și mergi încet = BAD
        stagnation_penalty = 0.0
        min_front_dist = 200.0
        
        if self.road:
            for v in self.road.vehicles:
                if v is self.vehicle: continue
                try: v_lane = v.lane_index[2]
                except: continue
                
                if v_lane == current_lane_idx:
                    d = v.position[0] - self.vehicle.position[0]
                    if 0 < d < min_front_dist:
                        min_front_dist = d
        
        # Penalizare STAGNARE - mai agresivă!
        # Se aplică dacă: mașină aproape (<30m) ȘI nu mergi la viteză maximă
        # Penalizare graduală: cu cât ești mai aproape și mai încet, cu atât e mai mare
        if min_front_dist < 30.0 and current_speed < max_speed * 0.9:
            # Factori: cât de aproape (1.0 la 5m, 0.0 la 30m) și cât de încet
            dist_factor = max(0.0, 1.0 - min_front_dist / 30.0)
            speed_factor = max(0.0, 1.0 - current_speed / max_speed)
            stagnation_penalty = dist_factor * (0.5 + speed_factor)  # 0.5 - 1.5
        
        # --- 3. CLEAR PATH ---
        # Bonus dacă ai drum liber (>40m) - încurajează să găsești spațiu
        clear_path = 1.0 if min_front_dist > 40.0 else 0.0

        # --- 4. DEPĂȘIRI ---
        overtaken_count = float(self.last_step_overtaken)
        normal_overtake = 0.0
        hero_overtake = 0.0
        
        if overtaken_count > 0:
            if is_oncoming_lane:
                hero_overtake = overtaken_count  # Depășire pe contrasens = JACKPOT
            else:
                normal_overtake = overtaken_count  # Depășire normală

        # --- 5. NEAR MISS (trecere pe lângă contrasens) ---
        raw_near_miss_count = float(self.last_step_oncoming_passed)
        final_near_miss_reward = 0.0
        if is_oncoming_lane > 0:
            final_near_miss_reward = raw_near_miss_count

        # --- 6. LANE CHANGE ---
        lane_change = 1.0 if action in [0, 2] else 0.0
        
        # --- 7. PROGRES PE DRUM ---
        # Bonus mic pentru progresul făcut (încurajează să meargă înainte)
        road_len = self.config.get("road_length", 1000)
        progress = self.vehicle.position[0] / road_len  # 0.0 -> 1.0
        progress_reward = progress  # Crește pe măsură ce avansezi

        # --- 8. FINALIZARE ---
        finish_bonus = 0.0
        duration = self.config.get("duration", 40)
        
        if self.vehicle.position[0] >= road_len - 10:
            # VICTORIE - bonus mare!
            finish_bonus = 50.0
        elif self.time >= duration - (1.0 / self.config["simulation_frequency"]):
            # TIMEOUT - PENALIZARE! Nu e ok să stai pe loc
            finish_bonus = self.config.get("timeout_penalty", -30.0)
            
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": speed_reward,
            "oncoming_lane_reward": is_oncoming_lane,
            "lane_change_reward": lane_change,
            "clear_path_reward": clear_path,
            "overtaking_reward": normal_overtake,
            "oncoming_overtake_reward": hero_overtake,
            "near_miss_reward": final_near_miss_reward, 
            "finish_reward": finish_bonus,
            "stagnation_penalty": stagnation_penalty,
            "progress_reward": progress,
            "same_lane_penalty": same_lane_penalty,  # NOU: penalizare pentru a sta pe aceeași bandă
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
        # Episodul se termină când ego ajunge aproape de capătul benzii SAU timpul a expirat.
        lane = self.vehicle.lane
        s, _ = lane.local_coordinates(self.vehicle.position)
        road_end = s >= lane.length - 5.0
        time_out = self.time >= self.config.get("duration", 40)
        
        if road_end:
            print(f"\n[Env] Truncated: Road End reached (s={s:.1f}/{lane.length})")
        elif time_out:
            print(f"\n[Env] Truncated: Time Out (t={self.time:.1f}/{self.config.get('duration', 40)})")
            
        return road_end or time_out


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
