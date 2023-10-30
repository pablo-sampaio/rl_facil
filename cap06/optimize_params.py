
import gymnasium as gym
import optuna

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap05.expected_sarsa import run_expected_sarsa
from cap05.qlearning_sarsa import run_qlearning

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from envs.wrappers import ObservationDiscretizerWrapper



# Esta função faz um treinamento com o Expected-SARSA, usando parâmetros sugeridos pelo Optuna.
# Retorna a média dos retornos dos últimos 100 episódios.
def train_with_exp_sarsa(trial : optuna.Trial):
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    lr = trial.suggest_float('learning_rate', 0.001, 1.0, log=True)
    #lr = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
    eps = trial.suggest_float('epsilon', 0.01, 0.2)
    bins1 = trial.suggest_int('bins1', 10, 100, step=10)
    bins2 = trial.suggest_int('bins2', 10, 100, step=10)
    
    print(f"\nTRIAL #{trial.number}: lr={lr}, eps={eps}, bins={bins1},{bins2}")

    env = gym.make("MountainCar-v0")

    # roda o algoritmo e recebe os retornos não-descontados
    env_wrapper = ObservationDiscretizerWrapper(env, [bins1,bins2])
    (returns, _) = run_expected_sarsa(env_wrapper, 2000, lr=lr, epsilon=eps)

    return sum(returns[-100:])/100 


def train_with_qlearning(trial : optuna.Trial):
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    lr = trial.suggest_uniform('learning_rate', 0.001, 1.0)
    #lr = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
    eps = trial.suggest_uniform('epsilon', 0.01, 0.2)
    bins1 = trial.suggest_int('bins1', 10, 100, step=10)
    bins2 = trial.suggest_int('bins2', 10, 100, step=10)
    
    print(f"\nTRIAL #{trial.number}: lr={lr}, eps={eps}, bins={bins1},{bins2}")

    env = gym.make("MountainCar-v0")

    # roda o algoritmo e recebe os retornos não-descontados
    env_wrapper = ObservationDiscretizerWrapper(env, [bins1,bins2])
    (returns, _) = run_qlearning(env_wrapper, 2000, lr=lr, epsilon=eps, render=False)

    return sum(returns[-100:])/100 


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///cap06//optuna_cap06.db', 
                            study_name= 'esarsa_results1', 
                            load_if_exists=True)
    
    # maximiza o valor de retorno de train_with_exp_sarsa, rodando 20 vezes
    # o parâmetro "n_jobs" indica a quantidade de CPUs a serem usadas (-1 para usar todas)
    study.optimize(train_with_exp_sarsa, n_trials=20, n_jobs=4) 

    print("MELHORES PARÂMETROS:")
    print(study.best_params)

