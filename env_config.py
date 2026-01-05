"""Config centralizat pentru environmentul two-way cu 4 benzi."""

# Lanes_count este pe direcție în TwoWayEnv (4 -> total 4 benzi în același sens, 2 sensuri)
ENV_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 75,
    "duration": 95,
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
