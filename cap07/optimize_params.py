
import numpy as np

import gymnasium as gym
from gymnasium.envs import register

import optuna

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap05.qlearning_sarsa import run_qlearning
from util.experiments import repeated_exec

#from envs import RacetrackEnv

# Registra o ambiente com o nome de "Racetrack"
register(
    id="Racetrack",  # Use a unique string ID for your environment
    entry_point="envs:RacetrackEnv",  # Specify the module and class name
    max_episode_steps=500,
)

# Variável para indicar o nome do ambiente a ter o Q-Learning otimizado
# Tente com "Racetrack" ou "Taxi-v3" ou "Frozen-Lake-v1" outro ambiente de estado discreto
ENV_NAME = "Racetrack"

EPISODES_PER_TRIAL = 800   # Se for o Taxi-v3, é suficiente usar por volta de 250
RUNS_PER_TRIAL     = 3     # Se for o Frozen-Lake, use por volta de 7


def train(trial : optuna.Trial):
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    lr = trial.suggest_float('lr', 0.1, 1.0)
    eps = trial.suggest_float('epsilon', 0.01, 0.2)
    gamma = trial.suggest_float('gamma', 0.5, 1.0)

    print(f"\nTRIAL #{trial.number}: lr={lr}, eps={eps}, gamma={gamma}")

    env = gym.make(ENV_NAME)
    results = repeated_exec(RUNS_PER_TRIAL, "qlearn-optuna", run_qlearning, env, EPISODES_PER_TRIAL, lr=lr, epsilon=eps, gamma=gamma)

    # soma dos retornos não-descontado finais (dos últimos 50 episódios)
    return np.sum(results[1][:,-50:])


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize',
                            storage='sqlite:///cap07//optuna_cap07.db',
                            study_name=f'qlearning_{ENV_NAME}',
                            load_if_exists=True)

    study.optimize(train, n_trials=40, n_jobs=4)

    print(f"MELHORES PARÂMETROS PARA {ENV_NAME}:")
    print(study.best_params)

