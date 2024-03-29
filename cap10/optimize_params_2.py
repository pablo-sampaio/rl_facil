
import gymnasium as gym
import optuna

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap10.actor_critic_nstep import run_vanilla_actor_critic_nstep
import cap09.models_torch_pg as models

ENV_NAME = "LunarLander-v2"
RUNS_PER_TRIAL = 3
STEPS_PER_RUN = 70_000

INPUTS = gym.make(ENV_NAME).observation_space.shape[0]
OUTPUTS = gym.make(ENV_NAME).action_space.n

models.DEFAULT_DEVICE = "cpu"


# Versão com exploração, otimizando fator de entropia da loss function da política
# Retorna a média dos retornos dos últimos 100 episódios
def train_actor_critic_with_entropy(trial : optuna.Trial):
    pol_lr      = trial.suggest_float('p_lr', 5e-6, 0.1, step=5e-6)
    rel_val_lr  = trial.suggest_float('relative_v_lr', 0.1, 10.0, step=0.1)
    nsteps      = trial.suggest_int('nsteps', 1, 64)
    expl_factor  = trial.suggest_float('expl_factor', 0.0, 1.0, step=2e-3)
    val_lr      = rel_val_lr * pol_lr

    print(f"\nTRIAL #{trial.number}: {trial.params}")

    env = gym.make(ENV_NAME)
    sum_results = 0.0

    for i in range(RUNS_PER_TRIAL):
        policy_model = models.PolicyModelPGWithExploration(INPUTS, [256,256], OUTPUTS, exploration_factor=expl_factor, lr=pol_lr)
        Vmodel = models.ValueModel(INPUTS, [256,256], lr=val_lr)
        
        returns, _ = run_vanilla_actor_critic_nstep(env, STEPS_PER_RUN, 0.99, nsteps=nsteps, initial_policy=policy_model, initial_v_model=Vmodel, verbose=False)
        
        return_100 = [ ret for (step, ret) in returns[-100:] ]  # pode ter menos de 100
        mean_return_100 = sum(return_100) / len(return_100)
        print(f"- trial #{trial.number}, run #{i+1} finished with {mean_return_100}")
        
        trial.report(mean_return_100, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        sum_results += mean_return_100

    return sum_results/RUNS_PER_TRIAL


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///cap10//optuna_cap10.db', 
                            study_name= 'ac_nstep_lunarlander_with_entropy', 
                            pruner=optuna.pruners.MedianPruner(),
                            load_if_exists=True)
    
    study.optimize(train_actor_critic_with_entropy, n_trials=30, n_jobs=4)

    print("MELHORES PARÂMETROS:")
    print(study.best_params)

