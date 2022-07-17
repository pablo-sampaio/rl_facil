
import gym
import optuna

from expected_sarsa import run_expected_sarsa
#from wrappers import DiscreteObservationWrapper


ENV = gym.make("Taxi-v3")


# Esta função faz um treinamento com o Expected-SARSA, usando parâmetros sugeridos pelo Optuna.
# Retorna a média dos retornos dos últimos 100 episódios.
def train_with_exp_sarsa(trial : optuna.Trial):
    
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    lr = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
    eps = trial.suggest_uniform('epsilon', 0.01, 0.2)
    print(f"\nTRIAL #{trial.number}: lr={lr}, eps={eps}")

    # roda o algoritmo e recebe os retornos não-descontados
    (returns, _) = run_expected_sarsa(ENV, 2000, lr=lr, epsilon=eps)

    return sum(returns[-100:])/100 


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///optuna_studies.db', 
                            study_name= 'expected_sarsa_results', 
                            load_if_exists=False)
    
    # maximiza o valor de retorno de train_with_exp_sarsa, rodando 20 vezes
    study.optimize(train_with_exp_sarsa, n_trials=20) 

    print("MELHORES PARÂMETROS:")
    print(study.best_params)

