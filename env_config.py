"""Config centralizat pentru environmentul two-way cu 4 benzi."""

# Lanes_count este pe direcție în TwoWayEnv (4 -> total 4 benzi în același sens, 2 sensuri)
ENV_CONFIG = {
    "lanes_count": 4,
    # Traffic spawning - RANDOMIZAT
    # Pentru mai mult trafic: crește min/max_initial_vehicles și traffic_density
    "min_initial_vehicles": 75,   # Minim mașini la start
    "max_initial_vehicles": 75,   # Maxim mașini la start  
    "traffic_density": 100,        # Țintă: câte mașini în zona vizibilă (~150m)
    "duration": 150,               # 150 secunde - suficient pentru 1250m la ~10m/s
    "controlled_vehicles": 1,
    "ego_spacing": 2,
    
    # --- REWARD TUNING pentru Q-Learning ---
    
    # 1. COLIZIUNE (Moartea)
    # Trebuie să fie destul de mare să descurajeze riscuri, dar nu să domine total
    "collision_reward": -50.0,

    # 2. VITEZĂ (per step) - MĂRIT pentru a încuraja viteza mare
    # La 750 steps cu viteză maximă: 0.2 × 750 = 150 puncte
    "high_speed_reward": 0.2,
    
    # 3. BONUSURI ACTIVE (Evenimente unice - PRINCIPALE!)
    
    # a. Depășire Normală - Recompensa principală!
    "overtaking_reward": 3.0, 
    
    # b. Depășire pe Contrasens - Risc mare = Reward mare, dar nu nebunesc
    "oncoming_overtake_reward": 5.0,
    
    # c. Near Miss (Trecere pe lângă mașini din contrasens)
    "near_miss_reward": 1.0,

    # 4. GHIDAJ (per step - MICI!)
    
    # a. Clear Path - Foarte mic, doar ghidaj
    "clear_path_reward": 0.05, 

    # b. Pe Contrasens (pasiv) - Mic, doar să încurajeze explorarea
    "oncoming_lane_reward": 0.1,

    # c. Schimbare Bandă - Cost mic de tranzacție
    "lane_change_reward": -0.1,
    
    # d. STAGNATION - Penalizare când stai blocat (MĂRIT!)
    "stagnation_penalty": -1.5,
    
    # e. PROGRESS - Încurajează să avanseze pe drum
    "progress_reward": 0.3,
    
    # f. FINISH - Bonus mare pentru victorie, PENALIZARE pentru timeout!
    "finish_reward": 1.0, # Multiplicator: 50 puncte victorie
    
    # g. TIMEOUT PENALTY - Penalizare când nu termini traseul!
    "timeout_penalty": -30.0,

    "offroad_terminal": True,
    "road_length": 1250,
   
    
    # Opțiuni de vizualizare; ppo.py le ignoră dacă nu randează
    "screen_width": 1300,
    "screen_height": 800,
    "scaling": 9,
    "centering_position": [0.25, 0.5],
    "manual_control": False,
    # 1. Intervalul pentru Reward (Normalizare)
    # Îi spunem agentului: "Viteza mică e 20, viteza maximă e 30".
    "reward_speed_range": [5, 15],

    # 2. Viteza Traficului (NPC)
    # Pentru gameplay mai lent, reduce această valoare (ex: 12-15)
    "other_vehicles_speed": 10.0, 

    # 3. Viteza de Start a Agentului
    "ego_init_speed": 10.0,
    
    # 4. Acțiunile Posibile (Vitezele Țintă)
    # Pentru joc mai lent, reduce aceste valori
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": True,
        # Viteze mai mici pentru gameplay mai lung:
        # 10 m/s = 36 km/h (Slow)
        # 15 m/s = 54 km/h (Normal) 
        # 20 m/s = 72 km/h (Fast)
        "target_speeds": [5, 10, 15], 
    },
    "simulation_frequency": 15,  # Mai puțini pași de fizică (era 15)
    "policy_frequency": 5,       # 5 acțiuni pe secundă (era 1) -> step() mult mai rapid
}
