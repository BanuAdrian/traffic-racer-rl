"""Config centralizat pentru environmentul two-way cu 4 benzi."""

# Lanes_count este pe direcție în TwoWayEnv (4 -> total 4 benzi în același sens, 2 sensuri)
ENV_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 100,
    "duration": 40,
    "controlled_vehicles": 1,
    "ego_spacing": 2,
    "collision_reward": -5.0,
    "high_speed_reward": 0.5,
    "reward_speed_range": [20, 45],
    "oncoming_lane_reward": 2.0, # Reward mare pentru mers pe contrasens
    "offroad_terminal": True,
    "road_length": 2000,
    
    # Opțiuni de vizualizare; ppo.py le ignoră dacă nu randează
    "screen_width": 1400,
    "screen_height": 800,
    "scaling": 30,
    "centering_position": [0.35, 0.55],
    "manual_control": False,
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": True,
        # Viteze țintă explicite (m/s): 0, 15, 30, 45. Pași mari (15 m/s ~ 54 km/h)
        "target_speeds": [0, 15, 30, 45],
    },
    "simulation_frequency": 15,  # Mai puțini pași pe secundă -> acțiuni mai "mari" per pas
    "policy_frequency": 1,
}
