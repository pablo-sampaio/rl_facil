
import gym
import optuna

from actor_critic_nstep import run_actor_critic_nstep, PolicyModelPGWithExploration, ValueModel

ENV = gym.make("CartPole-v1")
inputs = ENV.observation_space.shape[0]
outputs = ENV.action_space.n

RUNS_PER_TRIAL = 2

def train_actor_critic_nstep(trial : optuna.Trial):
    
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    nstep = trial.suggest_int('nstep', 1, 128)
    pol_lr = trial.suggest_loguniform('policy_lr', 1e-6, 1e-1)
    val_lr = trial.suggest_loguniform('value_lr', 1e-6, 1e-1)
    expl_factor = trial.suggest_uniform('expl_factor', 1e-3, 4.0)
    
    print(f"\nTRIAL #{trial.number}: {nstep=}, {pol_lr=}, {val_lr=}, {expl_factor=}")

    sum_results = 0.0

    for i in range(RUNS_PER_TRIAL):
        policy_model = PolicyModelPGWithExploration(inputs, [256,256], outputs, exploration_factor=expl_factor, lr=pol_lr)
        Vmodel = ValueModel(inputs, [256,32], lr=val_lr)
        returns, _ = run_actor_critic_nstep(ENV, 20000, 0.99, nstep=nstep, initial_policy=policy_model, initial_vmodel=Vmodel, verbose=False)
        mean_return_50 = sum(returns[-50:])/50.0
        
        trial.report(mean_return_50, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        sum_results += mean_return_50

    return sum_results/RUNS_PER_TRIAL


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///optuna_studies.db', 
                            study_name= 'actor_critic_nsteps_y', 
                            pruner=optuna.pruners.ThresholdPruner(lower=50.0),
                            load_if_exists=True)
    
    # maximiza o valor de retorno de "train_actor_critic_nstep", rodando-o várias vezes
    study.optimize(train_actor_critic_nstep, n_trials=20) 

    print("MELHORES PARÂMETROS:")
    print(study.best_params)

