
import gymnasium as gym
import optuna

from actor_critic_nstep import run_vanilla_actor_critic_nstep
from cap08.models_torch_pg import PolicyModelPGWithExploration, ValueModel

ENV = gym.make("CartPole-v1")
inputs = ENV.observation_space.shape[0]
outputs = ENV.action_space.n

RUNS_PER_TRIAL = 3

def train_actor_critic_nstep(trial : optuna.Trial):
    # precisa criar como variável local se for usar várias CPUs
    env = gym.make("CartPole-v1")
    
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    nsteps      = trial.suggest_int('nsteps', 1, 64)
    pol_lr      = trial.suggest_float('policy_lr', 1e-5, 0.1, log=True)
    val_lr      = trial.suggest_float('value_lr', 1e-5, 0.1, log=True)
    expl_factor = trial.suggest_float('expl_factor', 1e-3, 0.4)
    
    print(f"\nTRIAL #{trial.number}: {nsteps=}, {pol_lr=}, {val_lr=}, {expl_factor=}")

    sum_results = 0.0

    for i in range(RUNS_PER_TRIAL):
        policy_model = PolicyModelPGWithExploration(inputs, [256,256], outputs, exploration_factor=expl_factor, lr=pol_lr)
        Vmodel = ValueModel(inputs, [256,32], lr=val_lr)
        returns, _ = run_vanilla_actor_critic_nstep(env, 5_000, 0.99, nsteps=nsteps, initial_policy=policy_model, initial_v_model=Vmodel, verbose=False)
        mean_return_50 = sum([ ret for (step, ret) in returns[-50:] ])/50.0
        
        trial.report(mean_return_50, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        sum_results += mean_return_50

    return sum_results/RUNS_PER_TRIAL


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///optuna_studies.db', 
                            study_name='actor_critic_nsteps_10', 
                            pruner=optuna.pruners.MedianPruner(),
                            load_if_exists=True)
    
    # maximiza o valor de retorno de "train_actor_critic_nstep", rodando-o várias vezes
    # use n_jobs = -1 para usar todas as CPUs
    study.optimize(train_actor_critic_nstep, n_trials=40, n_jobs=8)

    print("MELHORES PARÂMETROS:")
    print(study.best_params)

