
import numpy as np
import gymnasium as gym
import optuna

import sys
from os import path
import argparse

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap05.qlearning_sarsa import run_qlearning
from util.experiments import repeated_exec

import envs  # importa o racetrack


# Default environment name
ENV_NAME = "RaceTrack-v0"

EPISODES_PER_TRIAL = 800   # Se for o Taxi-v3, é suficiente usar por volta de 250
RUNS_PER_TRIAL     = 3     # Se for o FrozenLake, use por volta de 7


def train(trial : optuna.Trial):
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    lr = trial.suggest_float('lr', 0.01, 1.0)
    eps = trial.suggest_float('epsilon', 0.01, 0.2)
    gamma = trial.suggest_float('gamma', 0.8, 1.0)

    print(f"\nTRIAL #{trial.number}: lr={lr}, eps={eps}, gamma={gamma}")

    env = gym.make(ENV_NAME)
    results = repeated_exec(RUNS_PER_TRIAL, "qlearn-optuna", run_qlearning, env, EPISODES_PER_TRIAL, lr=lr, epsilon=eps, gamma=gamma)

    # soma dos retornos não-descontado finais (dos últimos 50 episódios)
    return np.sum(results[1][:,-50:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Q-Learning parameters for a given environment.')
    parser.add_argument('--env', type=str, default=ENV_NAME, help='Name of the environment to optimize. Default is RaceTrack-v0.')
    
    args = parser.parse_args()

    ENV_NAME = args.env

    study = optuna.create_study(direction='maximize',
                                storage='sqlite:///optuna_cap07.db',
                                study_name=f'qlearning_{ENV_NAME}',
                                load_if_exists=True)

    study.optimize(train, n_trials=20, n_jobs=2)

    print(f"MELHORES PARÂMETROS PARA {ENV_NAME}:")
    print(study.best_params)

