"""Config centralizat pentru environmentul two-way cu 4 benzi."""

# Lanes_count este pe direcție în TwoWayEnv (4 -> total 4 benzi în același sens, 2 sensuri)
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
    # Opțiuni de vizualizare; ppo.py le ignoră dacă nu randează
    "screen_width": 1400,
    "screen_height": 800,
    "scaling": 30,
    "centering_position": [0.35, 0.55],
}
