
import gymnasium as gym
import optuna

import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap10.actor_critic_nstep import run_vanilla_actor_critic_nstep
#from cap10.actor_critic import run_vanilla_actor_critic


STEPS_PER_RUN = 80_000


# Retorna a média dos retornos dos últimos 100 episódios.
def train_actor_critic(trial : optuna.Trial):
    trial.suggest_float('p_lr', 1e-5, 1e-3)
    trial.suggest_float('relative_v_lr', 0.1, 10.0, step=0.1)
    trial.suggest_int('nsteps', 1, 64)

    print(f"\nTRIAL #{trial.number}: {trial.params}")

    env = gym.make("LunarLander-v2")

    # roda o algoritmo e recebe os retornos não-descontados
    (returns, _) = run_vanilla_actor_critic_nstep(env, STEPS_PER_RUN, gamma=0.99, **trial.params, verbose=False)

    return np.array(returns[-100:])[:,1].mean()


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///cap10//optuna_cap10.db', 
                            study_name= 'ac_nstep_lunarlander', 
                            load_if_exists=True)
    
    study.optimize(train_actor_critic, n_trials=20, n_jobs=4) 

    print("MELHORES PARÂMETROS:")
    print(study.best_params)

