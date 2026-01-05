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
        vehicles_count = self.config.get("vehicles_count", 80)
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # 1. Creăm EGO (Agentul)
        # Îl punem pe banda 1 (interior) sau 0 (exterior)
        ego_lane_idx = 1
        ego_vehicle = self.action_type.vehicle_class(
            road, 
            road.network.get_lane(("a", "b", ego_lane_idx)).position(30.0, 0.0), 
            speed=self.config.get("ego_init_speed", 20.0)
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        if vehicles_count <= 0:
            return

        # 2. Generăm Trafic NPC (Non-Uniform)
        # Nu mai folosim spacing fix. Încercăm să punem mașini random.
        
        spawned_count = 0
        attempts = 0
        max_attempts = vehicles_count * 5  # Evităm bucla infinită dacă e drumul plin
        
        min_spacing = 15.0 # Distanța minimă între mașini (buffer de siguranță la spawn)

        while spawned_count < vehicles_count and attempts < max_attempts:
            attempts += 1
            
            # Alege o poziție random pe tot drumul (după zona de start a agentului)
            x_pos = self.np_random.uniform(50, length - 20)
            
            # Alege o bandă random (0-3)
            if self.np_random.random() < 0.70: 
                # 70% șansă: Alegem benzile 0 sau 1 (Sensul Nostru)
                lane_choice = self.np_random.integers(0, 2)
            else:
                # 30% șansă: Alegem benzile 2 sau 3 (Contrasens)
                lane_choice = self.np_random.integers(2, 4)
            
            # Determinăm ID-ul benzii corecte în funcție de sens
            if lane_choice == 0: lane_id = ("a", "b", 0)
            elif lane_choice == 1: lane_id = ("a", "b", 1)
            elif lane_choice == 2: lane_id = ("b", "a", 0)
            else: lane_id = ("b", "a", 1)
            
            # --- SAFETY CHECK ---
            # Verificăm dacă locul e liber. Nu vrem spawn kills.
            is_safe = True
            for v in road.vehicles:
                # Verificăm doar mașinile de pe același sens/bandă logică
                # Atenție: lane_index e complex ("a", "b", 0). Verificăm tot tuplul sau doar indexul final?
                # Cel mai sigur: distanța Euclidiană. Dacă e vreo mașină prea aproape pe X și Y, skip.
                
                # Coordonatele propuse
                proposed_lane = road.network.get_lane(lane_id)
                proposed_pos = proposed_lane.position(x_pos, 0.0)
                
                # Distanța față de mașina existentă v
                dx = abs(v.position[0] - proposed_pos[0])
                dy = abs(v.position[1] - proposed_pos[1])
                
                # Dacă e pe aceeași bandă (dy mic) și prea aproape (dx mic)
                if dy < 2.0 and dx < min_spacing:
                    is_safe = False
                    break
            
            if is_safe:
                # Setăm viteza
                # Putem varia viteza puțin pentru realism
                base_speed = self.config.get("other_vehicles_speed", 20.0)
                speed = self.np_random.uniform(base_speed * 0.9, base_speed * 1.1)
                
                try:
                    lane_obj = road.network.get_lane(lane_id)
                    heading = lane_obj.heading_at(x_pos)
                    
                    v = vehicles_type(
                        road,
                        position=lane_obj.position(x_pos, 0.0),
                        heading=heading,
                        speed=speed,
                        enable_lane_change=False, # Traficul ține banda (mai sigur pt început)
                    )
                    road.vehicles.append(v)
                    spawned_count += 1
                except Exception:
                    pass # Ignorăm erori rare de geometrie

        self.last_step_overtaken = 0
        self.last_step_oncoming_passed = 0  # <--- INIȚIALIZARE NOUĂ

    def step(self, action: int):
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
        
        # Iterăm printr-o COPIE a listei ([:] este crucial!)
        # Nu putem șterge elemente dintr-o listă în timp ce iterăm prin original.
        for v in self.road.vehicles[:]:
            # Nu ștergem Ego-ul (Agentul)! El trebuie să ajungă la final ca să ia premiul.
            if v is not self.vehicle:
                # Ștergem mașinile care sunt în ultimii 10 metri de drum.
                # Dacă drumul e 1500, ștergem tot ce trece de 1490.
                # Astfel dispar ÎNAINTE să se oprească în zid.
                if v.position[0] >= road_len - 10.0:
                    # Îl scoatem din simulare -> Dispare instant
                    self.road.vehicles.remove(v)
                    
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
        Mixul Final:
        1. Viteză Pătratică (dependență de viteză maximă)
        2. Depășire Heroică (Bonus masiv pe contrasens)
        3. Clear Path (Anticipare trafic)
        """
        
        # --- 1. VITEZA PĂTRATICĂ ---
        # 25 m/s = 1.0 puncte
        # 20 m/s = 0.64 puncte
        # 10 m/s = 0.16 puncte
        # Asta elimină nevoia de "low_speed_penalty".
        max_speed = 25.0
        current_speed = self.vehicle.speed
        speed_reward = (current_speed / max_speed) ** 2
        
        # 1. Detectare dacă suntem pe contrasens
        current_lane_idx = 0
        try: current_lane_idx = self.vehicle.lane_index[2]
        except: pass
        
        # Ești pe contrasens dacă indexul e 2 sau 3
        is_oncoming_lane = 1.0 if current_lane_idx >= 2 else 0.0

        # --- LOGICA DE REWARD PENTRU INTERACȚIUNI ---
        
        # 1. Depășire Normală (Sensul meu)
        overtaken_count = float(self.last_step_overtaken)
        normal_overtake = 0.0
        hero_overtake = 0.0
        
        if overtaken_count > 0:
            if is_oncoming_lane:
                hero_overtake = overtaken_count # Depășesc stând pe contrasens (Foarte periculos!)
            else:
                normal_overtake = overtaken_count # Depășesc normal
        
        # --- MODIFICAREA AICI ---
        # 2. Near Miss (Trecere pe lângă trafic din față)
        raw_near_miss_count = float(self.last_step_oncoming_passed)
        final_near_miss_reward = 0.0
        
        # Condiția: Primești puncte DOAR dacă ești tu pe contrasens
        # în momentul în care treci pe lângă ei.
        if is_oncoming_lane > 0:
            final_near_miss_reward = raw_near_miss_count

        # --- 3. CLEAR PATH (Ochii din față) ---
        # Verificăm dacă avem drum liber pe 60m în față pe banda curentă
        clear_path = 0.0
        min_front_dist = 200.0
        
        if self.road:
            for v in self.road.vehicles:
                if v is self.vehicle: continue
                try: v_lane = v.lane_index[2]
                except: continue
                
                # Dacă e pe banda mea și în față
                if v_lane == current_lane_idx:
                    d = v.position[0] - self.vehicle.position[0]
                    if 0 < d < min_front_dist:
                        min_front_dist = d
        
        # Dacă am > 60m liberi, primesc un mic bonus constant
        if min_front_dist > 60.0:
            clear_path = 1.0

        # --- 4. LANE CHANGE ---
        lane_change = 1.0 if action in [0, 2] else 0.0

        # --- LOGICA DE FINALIZARE ---
        finish_bonus = 0.0
        
        road_len = self.config.get("road_length", 1000)
        duration = self.config.get("duration", 40)
        
        # 1. Verificăm ROAD END (Victorie Totală)
        # Am parcurs tot drumul -> Viteza a fost bună -> BONUS MARE
        if self.vehicle.position[0] >= road_len - 10:
            finish_bonus = 50.0  
            
        # 2. Verificăm TIME OUT (Supraviețuire)
        # Nu am ajuns la capăt, dar timpul e pe sfârșite -> BONUS MEDIU
        # self.time crește cu 1/FPS la fiecare pas.
        # Verificăm dacă suntem în ultima secundă
        elif self.time >= duration - (1.0 / self.config["simulation_frequency"]):
            # Îi dăm puncte că a supraviețuit, dar mai puține decât dacă ajungea la capăt.
            # Astfel, nu e tentat să meargă încet doar ca să treacă timpul.
            finish_bonus = 20.0
            
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": (self.vehicle.speed / 25.0) ** 2,
            "oncoming_lane_reward": is_oncoming_lane,
            "lane_change_reward": lane_change,
            "clear_path_reward": clear_path,
            
            "overtaking_reward": normal_overtake,
            "oncoming_overtake_reward": hero_overtake,
            
            # Folosim variabila filtrată
            "near_miss_reward": final_near_miss_reward, 
            "finish_reward": finish_bonus
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
