
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import argparse

def plot_rewards(log_file, title="Antrenament", window=50, output_file="training_plot.png"):
    if not os.path.exists(log_file):
        print(f"Nu gasesc fisierul {log_file}")
        return

    episodes = []
    rewards = []

    try:
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    episodes.append(int(row["episode"]))
                    rewards.append(float(row["reward"]))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Eroare la citirea fisierului: {e}")
        return

    if not episodes:
        print("Nu sunt date valide in fisier.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.3, color='gray', label='Reward per Episod')
    
    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        # Ajustam x-axis pentru media mobila
        plt.plot(episodes[window-1:], moving_avg, color='blue', linewidth=2, label=f'Medie Mobila ({window} ep)')
    
    plt.xlabel('Episod')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file)
    print(f"Grafic salvat ca {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Ploteaza reward-uri din CSV.")
    parser.add_argument("--log-file", type=str, required=True, help="Calea catre fisierul CSV cu rewards")
    parser.add_argument("--title", type=str, default="Antrenament", help="Titlul graficului")
    parser.add_argument("--window", type=int, default=50, help="Fereastra pentru media mobila")
    parser.add_argument("--output", type=str, default="training_plot.png", help="Numele fisierului de iesire")
    
    args = parser.parse_args()
    
    plot_rewards(args.log_file, args.title, args.window, args.output)

if __name__ == "__main__":
    main()
