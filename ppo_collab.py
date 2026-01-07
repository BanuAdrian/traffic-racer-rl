import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Importăm clasele din fișierele create
from traffic_env import EndlessOvertakeEnv
from ppo import PPOAgent

# Setăm mediul fără grafică (headless) pentru Colab
os.environ["SDL_VIDEODRIVER"] = "dummy"

def train_on_colab():
    env = EndlessOvertakeEnv(render_mode=None) # Important: None
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Folosim GPU dacă e disponibil
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}...")

    agent = PPOAgent(state_dim, action_dim, lr=3e-4)
    # Mutăm rețelele pe GPU
    agent.policy.to(device)
    agent.value_fn.to(device)

    EPOCHS = 300           # Mai multe epoci pentru Colab
    STEPS_PER_EPOCH = 2048 
    
    reward_history = []

    for epoch in range(EPOCHS):
        observations = []
        actions = []
        logps = []
        values = []
        rewards = []
        dones = []

        state, _ = env.reset()
        ep_reward = 0
        
        # --- Rollout ---
        for step in range(STEPS_PER_EPOCH):
            # Agentul pe GPU, datele trebuie să fie compatibile
            action, logp, value = agent.act(state) 
            # (Nota: funcția ta 'act' din ppo.py mută automat pe tensor, 
            # dar asigură-te că modelul e pe device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            observations.append(state)
            actions.append(action)
            logps.append(logp)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            ep_reward += reward
            state = next_state

            if done:
                reward_history.append(ep_reward)
                state, _ = env.reset()
                ep_reward = 0

        # --- Pregătire date (conversie la tensori pe GPU) ---
        # Aici e cheia vitezei pe Colab
        obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions).to(device)
        logp_tensor = torch.stack(logps).to(device)
        # Values sunt deja pe device din .act(), doar luăm .item() pentru calcul GAE pe CPU e ok, 
        # sau totul pe GPU. Pentru simplitate, lasă GAE pe CPU și mută rezultatul.
        
        values_list = [v.item() for v in values]
        next_value_t = agent.value_fn(torch.tensor(state, dtype=torch.float32).to(device))
        next_value = next_value_t.item()
        
        # GAE calculat clasic (lista)
        advantages_list, returns_list = agent.compute_gae(rewards, values_list, next_value, dones)
        
        # Mutăm rezultatele GAE pe GPU pentru update
        advantages = advantages_list.to(device)
        returns = returns_list.to(device)

        # --- Train ---
        agent.train(obs_tensor, actions_tensor, logp_tensor, advantages, returns)

        # Logging
        avg_rew = np.mean(reward_history[-10:]) if len(reward_history) > 0 else 0
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Avg Reward = {avg_rew:.2f}")

    # Salvare
    agent.save("traffic_ppo_colab.pth")
    print("✅ Model salvat!")
    
    # Plotting
    plt.plot(reward_history)
    plt.title("Antrenament PPO (Colab)")
    plt.xlabel("Episoade")
    plt.ylabel("Reward")
    plt.show()

# Rulează
train_on_colab()