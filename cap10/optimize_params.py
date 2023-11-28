
import gymnasium as gym
import optuna

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap10.actor_critic_nstep import run_vanilla_actor_critic_nstep
#from cap10.actor_critic import run_vanilla_actor_critic
import cap09.models_torch_pg as models


ENV_NAME = "LunarLander-v2"
STEPS_PER_RUN = 80_000

models.DEFAULT_DEVICE = "cpu"


# Retorna a média dos retornos dos últimos 100 episódios
def train_actor_critic(trial : optuna.Trial):
    trial.suggest_float('p_lr', 5e-6, 0.01, step=5e-6)
    trial.suggest_float('relative_v_lr', 0.1, 10.0, step=0.1)
    trial.suggest_int('nsteps', 1, 64)

    print(f"\nTRIAL #{trial.number}: {trial.params}")

    env = gym.make(ENV_NAME)

    # roda o algoritmo e recebe os retornos não-descontados
    (returns, _) = run_vanilla_actor_critic_nstep(env, STEPS_PER_RUN, gamma=0.99, **trial.params, verbose=False)

    return_100 = [ ret for (step, ret) in returns[-100:] ]  # pode ter menos de 100

    return sum(return_100) / len(return_100)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///cap10//optuna_cap10.db', 
                            study_name= 'ac_nstep_lunarlander', 
                            load_if_exists=True)
    
    study.optimize(train_actor_critic, n_trials=20, n_jobs=4) 

    print("MELHORES PARÂMETROS:")
    print(study.best_params)

