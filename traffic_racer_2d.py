"""
Endless Overtake Game - 9 Actions (Combined Steering + Gas/Brake)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, List

# Constante
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 900 # H = 900 makes game easier
CAR_WIDTH = 40
CAR_HEIGHT = 70
LANE_WIDTH = 80
ROAD_LEFT = 40

LANE_X = [
    ROAD_LEFT + LANE_WIDTH * 0.5,   # Lane 0 - contrasens exterior
    ROAD_LEFT + LANE_WIDTH * 1.5,   # Lane 1 - contrasens interior
    ROAD_LEFT + LANE_WIDTH * 2.5,   # Lane 2 - sens nostru interior
    ROAD_LEFT + LANE_WIDTH * 3.5,   # Lane 3 - sens nostru exterior
]

BLACK = (30, 30, 30)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
YELLOW = (255, 255, 0)
GREEN = (50, 200, 50)
BLUE = (50, 120, 220)
ORANGE = (255, 140, 0)
RED = (200, 50, 50)


class Car:
    def __init__(self, lane: int, y: float, speed: float, color: Tuple):
        self.lane = lane
        self.x = LANE_X[lane]
        self.y = y
        self.speed = speed
        self.color = color
    
    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.x - CAR_WIDTH // 2,
            self.y - CAR_HEIGHT // 2,
            CAR_WIDTH,
            CAR_HEIGHT
        )


class EndlessOvertakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Action space EXTINS la 9 actiuni pentru control fluid:
        # 0: Idle, 1: Accel, 2: Brake, 3: Left, 4: Right
        # 5: Accel+Left, 6: Accel+Right, 7: Brake+Left, 8: Brake+Right
        self.action_space = spaces.Discrete(9)
        
        # Observation: [ego_lane, ego_speed, x_offset, car1_lane, car1_rel_y, car1_rel_speed, ...]
        # 3 (ego) + 10 caars * 3 = 33
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32
        )
        
        # Config
        self.ego_y = SCREEN_HEIGHT * 0.7  # ego fix pe ecran
        self.min_speed = 10
        self.max_speed = 100
        self.same_dir_speed_range = (12, 45)   # albastru - mai lente
        self.oncoming_speed_range = (10, 27)   # portocaliu
        self.spawn_gap = 120
        self.target_cars_per_lane = 2
        
        # keep this from 1 - 10 i think
        self.traffic_density = 4
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Ego vehicle
        self.ego_lane = 2
        self.ego_speed = 30.0
        self.ego_x = LANE_X[self.ego_lane]
        self.move_speed = 4
        
        self.x_min = ROAD_LEFT + CAR_WIDTH // 2 + 5
        self.x_max = ROAD_LEFT + LANE_WIDTH * 4 - CAR_WIDTH // 2 - 5
        
        self.world_y = 0.0
        
        self.cars: List[Car] = []
        
        self.spawn_timer = 0
        self.active_pattern = None   # 'COLUMN', 'WALL', 'SINGLE' sau None
        self.pattern_counter = 0     # Câte mașini au mai rămas de spawnat în pattern
        self.pattern_lane = 0        # Pe ce bandă e pattern-ul curent
        
        self._spawn_car_on_lane(self.np_random.integers(0, 4), -300)
        
        self.steps = 0
        self.total_distance = 0
        self.overtakes = 0
        self.overtaken_cars = set()
        
        return self._get_obs(), {}
    
    
    def _maintain_traffic_density(self):
        self.spawn_timer -= 1
        
        if self.spawn_timer > 0:
            return

        spawn_y = -CAR_HEIGHT - 50

        speed_factor = 60.0 / max(self.ego_speed, 15.0)

        if self.active_pattern == 'COLUMN' and self.pattern_counter > 0:
            if self._is_lane_safe(self.pattern_lane, spawn_y):
                self._spawn_car_on_lane(self.pattern_lane, spawn_y)
                self.pattern_counter -= 1
                
                self.spawn_timer = 15
                
                if self.pattern_counter == 0:
                    self.active_pattern = None
                    base_wait = int(70 / self.traffic_density)
                    self.spawn_timer = int(base_wait * speed_factor)
            else:
                self.spawn_timer = 5
            return

        self.active_pattern = None
        
        roll = self.np_random.random()
        
        # 40% chance - SINGLE CAR
        if roll < 0.40:
            lane = self.np_random.integers(0, 4)
            if self._is_lane_safe(lane, spawn_y):
                self._spawn_car_on_lane(lane, spawn_y)
                base_wait = int(40 / self.traffic_density)
                self.spawn_timer = int(base_wait * speed_factor)
            else:
                self.spawn_timer = 5

        # 30% - WALL (ZID pe 2 benzi)
        elif roll < 0.70:
            start_lane = self.np_random.integers(0, 3)
            lane_a = start_lane
            lane_b = start_lane + 1
            
            if self._is_lane_safe(lane_a, spawn_y) and self._is_lane_safe(lane_b, spawn_y):
                self._spawn_car_on_lane(lane_a, spawn_y)
                self._spawn_car_on_lane(lane_b, spawn_y)
                
                base_wait = int(80 / self.traffic_density)
                self.spawn_timer = int(base_wait * speed_factor)
            else:
                self.spawn_timer = 5

        # 30% - TRAFFIC COLUMN
        else:
            lane = self.np_random.integers(0, 4)
            if self._is_lane_safe(lane, spawn_y):

                self.active_pattern = 'COLUMN'
                self.pattern_lane = lane
                self.pattern_counter = self.np_random.integers(2, 5) # 2-4 mașini
                
                self._spawn_car_on_lane(lane, spawn_y)
                self.pattern_counter -= 1
                self.spawn_timer = 15
            else:
                self.spawn_timer = 5

    def _spawn_car_on_lane(self, lane, y):
        is_oncoming = lane < 2
        color = ORANGE if is_oncoming else BLUE
        speed_range = self.oncoming_speed_range if is_oncoming else self.same_dir_speed_range
        speed = self.np_random.uniform(*speed_range)
        self.cars.append(Car(lane, y, speed, color))

    def _is_lane_safe(self, lane, y_pos):
        for car in self.cars:
            if car.lane == lane:
                if abs(car.y - y_pos) < CAR_HEIGHT * 1.5:
                    return False
        return True

    
    def _spawn_initial_traffic(self):
        pass

    def _spawn_new_car(self):
        pass

    # make sure cars dont "overlap" and instead form traffic "columns"
    def _enforce_car_spacing(self):
        min_dist = CAR_HEIGHT + 10
        
        for lane in range(4):
            lane_cars = [c for c in self.cars if c.lane == lane]
            
            if len(lane_cars) < 2:
                continue
            
            lane_cars.sort(key=lambda c: c.y)
            for i in range(len(lane_cars) - 1):
                front_car = lane_cars[i]
                back_car = lane_cars[i + 1]
                
                dist = back_car.y - front_car.y
                
                if dist < min_dist:
                    back_car.y = front_car.y + min_dist
    
    def _get_lane_from_x(self, x: float) -> int:
        for i in range(4):
            lane_left = ROAD_LEFT + LANE_WIDTH * i
            lane_right = ROAD_LEFT + LANE_WIDTH * (i + 1)
            if lane_left <= x < lane_right:
                return i
        return 3
    
    def step(self, action: int):
        self.steps += 1
        
        # === DECODIFICARE ACȚIUNI COMBINATE (Discrete 9) ===
        # 0: Idle
        # 1: Accel, 2: Brake, 3: Left, 4: Right
        # 5: Accel+Left, 6: Accel+Right, 7: Brake+Left, 8: Brake+Right
        
        want_accel = action in [1, 5, 6]
        want_brake = action in [2, 7, 8]
        want_left = action in [3, 5, 7]
        want_right = action in [4, 6, 8]

        # 1. Aplică Viteza (Longitudinal)
        if want_accel:
            self.ego_speed = min(self.ego_speed + 2.0, self.max_speed)
        elif want_brake:
            self.ego_speed = max(self.ego_speed - 3.0, self.min_speed)
        else:
            # Frecare naturală (nici acceleratie, nici frana)
            self.ego_speed = max(self.ego_speed - 0.5, self.min_speed)  # Coasting friction

        # 2. Aplică Direcția (Lateral)
        if want_left:
            self.ego_x = max(self.ego_x - self.move_speed, self.x_min)
        elif want_right:
            self.ego_x = min(self.ego_x + self.move_speed, self.x_max)
        
        # Calculează banda curentă bazat pe poziția X
        self.ego_lane = self._get_lane_from_x(self.ego_x)
        
        # === Update world ===
        self.world_y += self.ego_speed * 0.5
        self.total_distance += self.ego_speed
        
        # === Update mașini (relativ la ego) ===
        for car in self.cars:
            is_oncoming = car.lane < 2
            if is_oncoming:
                # Contrasens: vin spre noi (viteza relativă = ego + car)
                car.y += (self.ego_speed + car.speed) * 0.15
            else:
                # Același sens: noi îi depășim (viteza relativă = ego - car)
                car.y += (self.ego_speed - car.speed) * 0.15
            
            # === MODIFICARE: PREVENIRE IMPACT DIN SPATE ===
            # Dacă mașina e pe benzile noastre (2 sau 3)
            if car.lane >= 2:
                # Dacă e pe aceeași bandă cu noi
                if car.lane == self.ego_lane:
                    # Dacă e în spatele nostru (Y mai mare) și destul de aproape
                    if car.y > self.ego_y and (car.y - self.ego_y) < (CAR_HEIGHT + 30):
                        # Dacă are viteză mai mare decât noi, o egalăm (frânează)
                        if car.speed > self.ego_speed:
                            car.speed = self.ego_speed

        # === Menține distanța minimă între mașini pe aceeași bandă ===
        self._enforce_car_spacing()
        
        # === Elimină mașini care au ieșit de pe ecran ===
        self.cars = [c for c in self.cars if c.y < SCREEN_HEIGHT + 100]
        
        # === Spawn mașini noi (NOUA LOGICĂ) ===
        self._maintain_traffic_density()
        
        # === Verifică coliziuni ===
        ego_rect = pygame.Rect(
            self.ego_x - CAR_WIDTH // 2,
            self.ego_y - CAR_HEIGHT // 2,
            CAR_WIDTH,
            CAR_HEIGHT
        )
        
        collision = False
        for car in self.cars:
            if ego_rect.colliderect(car.get_rect()):
                collision = True
                break
        
        # === Calculează reward ===
        reward = self._calculate_reward(collision)
        
        # === Verifică terminare ===
        terminated = collision
        truncated = self.steps >= 3000  # max steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _calculate_reward(self, collision: bool) -> float:
        if collision:
            return -10.0
        
        reward = 0.0
        
        # speed bonus
        reward += (self.ego_speed / self.max_speed) * 0.5
        
        # crazy, risky driving bonus
        if self.ego_x <= LANE_X[1]:
            reward += 0.1
        
        # overtake bonus
        for car in self.cars:
            if car.lane >= 2:
                car_id = id(car)
                if car.y > self.ego_y + 20 and car_id not in self.overtaken_cars:
                    self.overtaken_cars.add(car_id)
                    self.overtakes += 1
                    reward += 1.0
        
        return reward
    
    def _get_obs(self) -> np.ndarray:
        """Observații: ego info + cele mai apropiate 10 mașini."""
        # Poziția X normalizată pe drum (0 = stânga, 1 = dreapta)
        x_norm = (self.ego_x - ROAD_LEFT) / (LANE_WIDTH * 4)
        
        obs = [
            x_norm,  # poziția X normalizată
            self.ego_speed / self.max_speed,
            self.ego_lane / 3.0,  # banda curentă
        ]
        
        # Sortează mașinile după distanță
        sorted_cars = sorted(self.cars, key=lambda c: abs(c.y - self.ego_y))
        
        # Ia primele 10
        for i in range(10):
            if i < len(sorted_cars):
                car = sorted_cars[i]
                obs.extend([
                    (car.x - ROAD_LEFT) / (LANE_WIDTH * 4),  # X normalizat
                    (car.y - self.ego_y) / 500.0,  # distanță relativă Y
                    car.speed / self.max_speed,
                ])
            else:
                obs.extend([0, 0, 0])  # padding
        
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Endless Overtake")
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        # Background
        self.screen.fill(GRAY)
        
        # Drum
        road_rect = pygame.Rect(ROAD_LEFT, 0, LANE_WIDTH * 4, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, BLACK, road_rect)
        
        # Linii
        line_y_offset = int(self.world_y * 2) % 40
        
        # Margini (continue)
        pygame.draw.line(self.screen, WHITE, (ROAD_LEFT, 0), (ROAD_LEFT, SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, WHITE, (ROAD_LEFT + LANE_WIDTH * 4, 0), 
                        (ROAD_LEFT + LANE_WIDTH * 4, SCREEN_HEIGHT), 3)
        
        # Linie centrală (continuă galbenă - separare sensuri)
        pygame.draw.line(self.screen, YELLOW, 
                        (ROAD_LEFT + LANE_WIDTH * 2, 0),
                        (ROAD_LEFT + LANE_WIDTH * 2, SCREEN_HEIGHT), 4)
        
        # Linii punctate între benzi
        for lane_sep in [1, 3]:  # între 0-1 și 2-3
            x = ROAD_LEFT + LANE_WIDTH * lane_sep
            for y in range(-40 + line_y_offset, SCREEN_HEIGHT, 40):
                pygame.draw.line(self.screen, WHITE, (x, y), (x, y + 20), 2)
        
        # Desenează mașinile
        for car in self.cars:
            is_oncoming = car.lane < 2
            self._draw_car(car.x, car.y, car.color, facing_down=is_oncoming)
        
        # Desenează ego (mereu în sus)
        self._draw_car(self.ego_x, self.ego_y, GREEN, facing_down=False)
        
        # HUD
        speed_text = self.font.render(f"Speed: {self.ego_speed:.0f} km/h", True, WHITE)
        self.screen.blit(speed_text, (10, 10))
        
        # Speed bar
        bar_width = 150
        bar_height = 15
        bar_x = 10
        bar_y = 45
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        fill_width = int((self.ego_speed / self.max_speed) * bar_width)
        bar_color = GREEN if self.ego_speed < 40 else YELLOW if self.ego_speed < 55 else RED
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height))
        
        overtake_text = self.font.render(f"Overtakes: {self.overtakes}", True, WHITE)
        self.screen.blit(overtake_text, (10, 70))
        
        lane_name = ["CONTRA-L", "CONTRA-R", "SENS-L", "SENS-R"][self.ego_lane]
        on_contra = "⚠️" if self.ego_x <= LANE_X[1] else ""
        lane_text = self.font.render(f"Lane: {lane_name} {on_contra}", True, WHITE)
        self.screen.blit(lane_text, (10, 105))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.array(pygame.surfarray.array3d(self.screen))
    
    def _draw_car(self, x: float, y: float, color: Tuple, facing_down: bool = False):
        """Desenează o mașină. facing_down=True pentru contrasens."""
        rect = pygame.Rect(
            x - CAR_WIDTH // 2,
            y - CAR_HEIGHT // 2,
            CAR_WIDTH,
            CAR_HEIGHT
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        
        # Parbriz (negru) - poziție diferită pentru contrasens
        if facing_down:
            # Contrasens: parbrizul jos (spre ego)
            windshield = pygame.Rect(
                x - CAR_WIDTH // 2 + 6,
                y + CAR_HEIGHT // 2 - 28,
                CAR_WIDTH - 12,
                20
            )
        else:
            # Sens normal: parbrizul sus
            windshield = pygame.Rect(
                x - CAR_WIDTH // 2 + 6,
                y - CAR_HEIGHT // 2 + 8,
                CAR_WIDTH - 12,
                20
            )
        pygame.draw.rect(self.screen, BLACK, windshield, border_radius=4)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# === DEMO FUNCTIONS ===

def demo_random(steps=1000):
    """Demo cu acțiuni aleatorii."""
    print("=" * 40)
    print("DEMO - Acțiuni aleatorii")
    print("🟢 Verde = Tu")
    print("🔵 Albastru = De depășit")
    print("🟠 Portocaliu = Contrasens")
    print("=" * 40)
    
    pygame.init()
    env = EndlessOvertakeEnv(render_mode="human")
    obs, _ = env.reset()
    env.render()  # Initialize display first
    
    total_reward = 0
    
    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        
        if terminated:
            print(f"💥 Coliziune! Step: {step}, Reward: {total_reward:.1f}, Overtakes: {env.overtakes}")
            obs, _ = env.reset()
            total_reward = 0
        
        if truncated:
            print(f"✅ Episod complet! Reward: {total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0
    
    env.close()


def demo_manual():
    """Demo cu control manual (tastatură)."""
    print("=" * 40)
    print("CONTROL MANUAL (9 ACTIONS)")
    print("Arrows + Combinations work now!")
    print("ESC = Ieși")
    print("=" * 40)
    
    pygame.init()
    env = EndlessOvertakeEnv(render_mode="human")
    obs, _ = env.reset()
    env.render()  # Initialize display first
    
    total_reward = 0
    running = True
    
    while running:
        action = 0  # implicit: nimic (0)
        
        # Input tastatură
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        
        # MAPPING 9 Actions:
        # 0: Idle
        # 1: Accel, 2: Brake, 3: Left, 4: Right
        # 5: Accel+Left, 6: Accel+Right, 7: Brake+Left, 8: Brake+Right
        
        if up:
            if left:
                action = 5
            elif right:
                action = 6
            else:
                action = 1
        elif down:
            if left:
                action = 7
            elif right:
                action = 8
            else:
                action = 2
        elif left:
            action = 3
        elif right:
            action = 4
        else:
            action = 0
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated:
            print(f"💥 Coliziune! Reward: {total_reward:.1f}, Overtakes: {env.overtakes}")
            import time
            time.sleep(1)
            obs, _ = env.reset()
            total_reward = 0
        
        if truncated:
            print(f"✅ Episod complet! Reward: {total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        demo_manual()
    else:
        demo_random()