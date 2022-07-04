
from time import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from bandit_envs import MultiArmedBandit


def run_epsilon_greedy(problem, total_steps, epsilon):
    num_arms = problem.get_num_arms()

    # stats per arm
    arm_prob = [0.0] * num_arms
    arm_n    = [0] * num_arms

    steps = 0
    rewards = np.empty(total_steps)
    problem.reset()

    for i in range(total_steps):
        if (np.random.random() <= epsilon):
            j = np.random.randint(num_arms)
        else:
            j = np.argmax(arm_prob)
        
        r = problem.step(j)
        steps += 1
        rewards[i] = r
        
        # update stats per arm
        arm_n[j] += 1
        arm_prob[j] = ((arm_n[j]-1) * arm_prob[j] + r) / arm_n[j]

    return (arm_prob, arm_n, rewards, f"Eps-greedy ({epsilon})")


def run_optmistic_init(problem, total_steps, p_init=1.0):
    num_arms = problem.get_num_arms()

    # stats per arm
    arm_prob = [p_init] * num_arms
    arm_n    = [0] * num_arms

    steps = 0
    rewards = np.empty(total_steps)
    problem.reset()

    for i in range(total_steps):
        j = np.argmax(arm_prob)
        
        r = problem.step(j)
        steps += 1
        rewards[i] = r
        
        # update stats per arm
        arm_n[j] += 1
        arm_prob[j] = (arm_n[j] * arm_prob[j] + r) / (arm_n[j]+1) # slightly changed to properly deal with the 1st sample

    return (arm_prob, arm_n, rewards, f"Opt-init ({p_init})")


def run_ucb(problem, total_steps):
    num_arms = problem.get_num_arms()

    # stats per arm
    arm_prob = np.zeros(num_arms)
    arm_n    = np.zeros(num_arms)

    steps = 0
    rewards = np.empty(total_steps)
    problem.reset()

    for j in range(num_arms):
        r = problem.step(j)
        steps += 1
        arm_n[j] += 1
        arm_prob[j] = ((arm_n[j]-1) * arm_prob[j] + r) / arm_n[j]
  
    for i in range(num_arms, total_steps):
        # inside the argmax: arithmetical operations on np arrays, resulting in a np array
        j = np.argmax( arm_prob + np.sqrt(2*np.log(steps) / arm_n) )
        
        r = problem.step(j)
        steps += 1
        rewards[i] = r

        # update stats per arm
        arm_n[j] += 1
        arm_prob[j] = ((arm_n[j]-1) * arm_prob[j] + r) / arm_n[j]

    return (arm_prob, arm_n, rewards, "UCB")


class ThompsonArmInfo:
    def __init__(self):
        self.a = 1
        self.b = 1
        self.N = 0 # for information only

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, r):
        self.a += r
        self.b += 1 - r
        self.N += 1
    
    def prob_mean(self):
        return beta.mean(self.a, self.b)

def plot_betas(problem, arminfos, trial):
    x = np.linspace(0, 1, 200)
    for i in range(len(arminfos)):
        b = arminfos[i]
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label=f"real p: {problem.probs[i]:.4f}, win rate = {b.a - 1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()

# only works with the win-loose version (MultiArmedBandit)
def run_thompson(problem, total_steps, show_distributions=False):
    num_arms = problem.get_num_arms()

    # keeps stats per "arm"
    arm_info = [ThompsonArmInfo() for i in range(num_arms)]

    rewards = np.empty(total_steps)
    problem.reset()

    #sample_points = [5,10,20,50,100,200,500,1000,1500,2000]
    sample_points = [10, 30, 1000, 5000, 50000, 100000]

    for step_i in range(total_steps):
        # Thompson sampling
        j = np.argmax( [a.sample() for a in arm_info] )

        r = problem.step(j)
        rewards[step_i] = r

        arm_info[j].update(r)

        # plot the posteriors
        if show_distributions and step_i+1 in sample_points:
            plot_betas(problem, arm_info, step_i+1)

    arm_prob = [a.prob_mean() for a in arm_info]
    arm_n =  [a.N for a in arm_info]
    return (arm_prob, arm_n, rewards, "Thompson")


def plot_results(results, best_prob):
    total_steps = results[0][2].shape[0]
    bestline = np.ones(total_steps) * best_prob
    bestline_style = "--"  # (0, (1, 1))
    bestline_color = "gray"
    
    # plot moving average ctr    
    for (_, _, rewards, alg_name) in results:
        cumulative_average = np.cumsum(rewards) / (np.arange(1, total_steps+1))
        plt.plot(cumulative_average, label=alg_name)
    plt.plot(bestline, linestyle=bestline_style, color=bestline_color)
    plt.xscale('log')
    plt.title(f"Cumulative average - x in log scale")
    plt.legend()
    plt.show()

    # plot moving average ctr, linear scale
    for (arm_prob, arm_n, rewards, alg_name) in results:
        cumulative_average = np.cumsum(rewards) / (np.arange(1, total_steps+1))
        plt.plot(cumulative_average, label=alg_name)
    plt.plot(bestline, linestyle=bestline_style, color=bestline_color)
    plt.title(f"Cumulative average")
    plt.legend()
    plt.show()

    for (arm_prob, arm_n, rewards, alg_name) in results:
        print("Summary for " + alg_name)
        print(" - total reward:", rewards.sum())
        print(" - avg reward (win rate):", rewards.sum() / total_steps)
        print(" - estimate means (probabilities):", arm_prob)
        print(" - arms selected:", arm_n)
        print()

    return

def repeated_exec(executions, algorithm, *args):
    mab_problem = args[0]
    num_steps = args[1]
    arm_prob = np.zeros(shape=(executions, mab_problem.get_num_arms()))
    arm_n = np.zeros(shape=(executions, mab_problem.get_num_arms()))
    rewards = np.zeros(shape=(executions, num_steps))
    alg_name = ""
    t = time()
    print(f"Executing {algorithm}:")
    for i in tqdm(range(executions)):
        arm_prob[i], arm_n[i], rewards[i], alg_name = algorithm(*args)
    t = time() - t
    print(f"  ({executions} executions of {alg_name} finished in {t:.2f} secs)")
    return arm_prob.mean(axis=0), arm_n.mean(axis=0), rewards.mean(axis=0), alg_name


if __name__ == '__main__':
    EXECUTIONS = 500
    NUM_STEPS = 100000
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
    mab_problem = MultiArmedBandit(BANDIT_PROBABILITIES)

    results = []

    results.append( repeated_exec(EXECUTIONS, run_epsilon_greedy, mab_problem, NUM_STEPS, 0.1) )
    results.append( repeated_exec(EXECUTIONS, run_optmistic_init, mab_problem, NUM_STEPS) )
    results.append( repeated_exec(EXECUTIONS, run_ucb, mab_problem, NUM_STEPS) )
    #results.append( repeated_exec(EXECUTIONS, run_thompson, mab_problem, NUM_STEPS) )
    
    #to run directly an algorithm only 1 time, do like this:
    #results.append( run_epsilon_greedy(mab_problem, NUM_STEPS, 0.1) ) 

    plot_results(results, mab_problem.get_best_mean_reward())
    