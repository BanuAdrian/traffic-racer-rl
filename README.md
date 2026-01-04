# Traffic Racer RL

Proiect de Reinforcement Learning folosind Q-Learning tabular pentru a invata un agent sa conduca in stil "Traffic Racer" (evitare trafic, mers pe contrasens).

## Cerinte

Instaleaza dependintelele:
```bash
pip install -r requirements.txt
```

## Utilizare

### 1. Control Manual
Pentru a testa environment-ul si fizica jocului controland masina din tastatura (sageti):

```bash
python run_env.py --manual
```

### 2. Antrenare Agent (Q-Learning)
Pentru a antrena agentul de la zero. Poti ajusta numarul de episoade.

```bash
python q_learning.py --episodes 100 --eval-render
```
* `--episodes`: Numarul de episoade de antrenament.
* `--eval-render`: Activeaza vizualizarea in timpul evaluarii (dupa antrenament).
* `--model-path`: Calea unde se salveaza tabelul Q (default: q_table.npy).

Daca intrerupi antrenamentul cu `CTRL+C`, progresul se salveaza automat.

### 3. Testare Agent (Evaluare)
Pentru a rula un agent deja antrenat fara a-l mai antrena (doar vizualizare):

```bash
python q_learning.py --eval-only --eval-episodes 10 --eval-render --model-path q_table.npy
```

## Configurare
Setarile environment-ului (densitate trafic, reward-uri, viteze) se gasesc in `env_config.py`.