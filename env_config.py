"""Config centralizat pentru environmentul two-way cu 4 benzi."""

# Lanes_count este pe direcție în TwoWayEnv (4 -> total 4 benzi în același sens, 2 sensuri)
ENV_CONFIG = {
    "lanes_count": 4,
    # Traffic spawning - RANDOMIZAT
    # Pentru mai mult trafic: crește min/max_initial_vehicles și traffic_density
    "min_initial_vehicles": 75,   # Minim mașini la start
    "max_initial_vehicles": 75,   # Maxim mașini la start  
    "traffic_density": 100,        # Țintă: câte mașini în zona vizibilă (~150m)
    "duration": 120,
    "controlled_vehicles": 1,
    "ego_spacing": 2,
    
    # --- REWARD TUNING ---
    
    # 1. SUPRAVIEȚUIRE (Costul Morții)
    # Mărim la -100. De ce?
    # Dacă un episod bun are 200 de puncte, -50 e acceptabil ca sacrificiu.
    # -100 înseamnă "Game Over" psihologic pentru agent.
    "collision_reward": -100.0,

    # 2. FLUXUL PASIV (Salariul de bază)
    # Îl ținem mic. Agentul nu trebuie să se îmbogățească doar stând degeaba.
    # Maxim 0.5 puncte/pas la viteză maximă.
    "high_speed_reward": 0.5,
    
    # 3. BONUSURI ACTIVE (Evenimente - Aici sunt banii!)
    
    # a. Depășire Normală
    # Trebuie să acopere costul schimbării de bandă și să dea profit.
    # Dacă schimbarea e -0.3, depășirea trebuie să fie semnificativă.
    "overtaking_reward": 2.0, 
    
    # b. Depășire Heroică (Pe Contrasens)
    # JACKPOT-ul. Vrem ca agentul să vadă contrasensul ca pe o mină de aur.
    # Valoarea 10.0 este imensă. Îl va face să ignore frica.
    "oncoming_overtake_reward": 10.0,
    
    # c. Near Miss (Trecere razantă pe lângă trafic din față)
    # Flux constant de dopamină când ești pe contrasens.
    # 1.5 puncte per mașină. Dacă trec 3 mașini într-o secundă = 4.5 puncte.
    "near_miss_reward": 1.5,

    # 4. GHIDAJ & CORECTURI (Busola)
    
    # a. Clear Path (Viziune)
    # Mic, doar cât să diferențieze o bandă liberă de una blocată.
    "clear_path_reward": 0.1, 

    # b. Stat pe Contrasens (Pasiv)
    # REDUCEM la 0.2. 
    # De ce? Nu vrem să stea pe contrasens când e gol (campare).
    # Vrem să stea pe contrasens DOAR ca să facă "Near Miss" sau "Hero Overtake".
    "oncoming_lane_reward": 0.2,

    # c. Schimbare Bandă (Costul de tranzacție)
    # Mărim puțin penalizarea la -0.3.
    # Elimină "tremuratul" inutil. Schimbi doar dacă ai un motiv (depășire).
    "lane_change_reward": -0.3,
    
    "finish_reward": 1.0, # Multiplicator (Logica de 20 vs 50 e în env)

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
