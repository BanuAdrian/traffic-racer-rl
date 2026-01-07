import torch
import pygame
import time
from traffic_racer_2d import EndlessOvertakeEnv
from tr2d_ppo import PPOAgent

def test():
    print("📺 Loading Trained Agent...")
    
    # 1. Creăm mediul CU GRAFICĂ
    env = EndlessOvertakeEnv(render_mode="human")
    pygame.init()
    env.render()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 2. Inițializăm agentul și încărcăm greutățile
    agent = PPOAgent(state_dim, action_dim)
    try:
        agent.load("traffic_ppo_colab.pth")
        # agent.load("traffic_ppo_model.pth")
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model not found! Run train.py first.")
        return

    # 3. Rulăm simularea
    num_episodes = 5
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"▶️ Starting Episode {ep+1}...")
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = agent.get_best_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            env.render()
            
            time.sleep(0.01) 

        print(f"🏁 Episode finished. Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test()