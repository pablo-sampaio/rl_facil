
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


# Default settings for the optimization
ENVIRONMENT_NAME   = "RaceTrack-v0"
RUNS_PER_TRIAL     = 3     # Se for o FrozenLake, use por volta de 7
EPISODES_PER_TRIAL = 800   # Se for o Taxi-v3, é suficiente usar por volta de 250


def train(trial : optuna.Trial):
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    lr = trial.suggest_float('lr', 0.01, 1.0)
    eps = trial.suggest_float('epsilon', 0.01, 0.2)
    gamma = trial.suggest_float('gamma', 0.8, 1.0)

    print(f"\nTRIAL #{trial.number}: lr={lr}, eps={eps}, gamma={gamma}")

    env = gym.make(ENVIRONMENT_NAME)
    results = repeated_exec(RUNS_PER_TRIAL, "qlearn-optuna", run_qlearning, env, EPISODES_PER_TRIAL, lr=lr, epsilon=eps, gamma=gamma)

    # soma dos retornos não-descontados finais (dos últimos 50 episódios)
    return np.sum(results[1][:,-50:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Q-Learning parameters for a given environment.')
    parser.add_argument('--env'               , type=str, default=ENVIRONMENT_NAME, help='Name of the environment to optimize. Default is RaceTrack-v0.')
    parser.add_argument('--runs_per_trial'    , type=int, default=RUNS_PER_TRIAL, help='Number of runs of Q-Learning algorithm executed per trial of the optimizer. Default is 3.')
    parser.add_argument('--episodes_per_trial', type=int, default=EPISODES_PER_TRIAL, help='Number of episodes used per run of the Q-Learning algorithm. Default is 800.')
    parser.add_argument('--trials'            , type=int, default=20, help='Number of trials to start the optimization. Default is 20.')
    
    args = parser.parse_args()

    ENVIRONMENT_NAME = args.env
    RUNS_PER_TRIAL = args.runs_per_trial
    EPISODES_PER_TRIAL = args.episodes_per_trial

    TRIALS = args.trials

    study = optuna.create_study(direction='maximize',
                                storage='sqlite:///optuna_cap07.db',
                                study_name=f'qlearning_{ENVIRONMENT_NAME}',
                                load_if_exists=True)

    study.optimize(train, n_trials=TRIALS, n_jobs=2)

    print(f"MELHORES PARÂMETROS PARA {ENVIRONMENT_NAME}:")
    print(study.best_params)
