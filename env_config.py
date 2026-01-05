"""Config centralizat pentru environmentul two-way cu 4 benzi."""

# Lanes_count este pe direcție în TwoWayEnv (4 -> total 4 benzi în același sens, 2 sensuri)
ENV_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 75,
    "duration": 50,
    "controlled_vehicles": 1,
    "ego_spacing": 2,
    
    # --- REWARD TUNING ---
    
    # 1. Supraviețuire
    "collision_reward": -50.0,

    # 2. Obiective Principale
    "high_speed_reward": 0.5,         # Viteză (Pătratică) 
       
    # a. Depășire Normală
    "overtaking_reward": 1.0,
    # b. Depășire Heroică (Eu sunt pe contrasens și depășesc pe unul de pe sensul meu)
    "oncoming_overtake_reward": 4.0, 
    # c. Near Miss (Trec pe lângă o mașină care vine din față)
    # Deoarece viteza relativă e mare, vei trece pe lângă multe mașini.
    # 0.5 puncte per mașină e un flux constant de dopamină pentru agent.
    "near_miss_reward": 0.5,

    # 3. Comportament Inteligent
    "clear_path_reward": 0.2,         # (Anticipare) E mic, dar constant.
                                      # Îl face să schimbe banda ÎNAINTE de obstacol.

    # 4. Ajustări Fine
    "oncoming_lane_reward": 0.5,      # Mic bonus pasiv pentru contrasens
    "lane_change_reward": -0.2,       # Penalizare zig-zag

    "offroad_terminal": True,
    "road_length": 1500,
   
    
    # Opțiuni de vizualizare; ppo.py le ignoră dacă nu randează
    "screen_width": 1300,
    "screen_height": 600,
    "scaling": 12,
    "centering_position": [0.25, 0.5],
    "manual_control": False,
    # 1. Intervalul pentru Reward (Normalizare)
    # Îi spunem agentului: "Viteza mică e 20, viteza maximă e 30".
    "reward_speed_range": [15, 25],

    # 2. Viteza Traficului (NPC)
    # Îl punem la 25 m/s (90 km/h) ca să fie provocator, dar depășibil.
    "other_vehicles_speed": 20.0, 

    # 3. Viteza de Start a Agentului
    # Plecăm direct lansat, ca să nu pierdem timp accelerând de la 0.
    "ego_init_speed": 20.0,
    
    # 4. Acțiunile Posibile (Vitezele Țintă)
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": True,
        # SCHIMBARE MAJORĂ:
        # 20 m/s = 72 km/h (Safe / Cruise)
        # 25 m/s = 90 km/h (Normal)
        # 30 m/s = 108 km/h (Fast / Overtake)
        "target_speeds": [15, 20, 25], 
    },
    "simulation_frequency": 15,  # Mai puțini pași de fizică (era 15)
    "policy_frequency": 5,       # 5 acțiuni pe secundă (era 1) -> step() mult mai rapid
}
