import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TorchMultiLayerNetwork(nn.Module):
    def __init__(self, input_dim, list_hidden_dims, output_dim, final_activ_fn=None):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in list_hidden_dims:
            layers.append( nn.Linear(last_dim, dim, bias=True) )
            layers.append( nn.ReLU() )
            last_dim = dim
        layers.append( nn.Linear(last_dim, output_dim, bias=True) )
        if final_activ_fn is not None:
            layers.append( final_activ_fn )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


# approximates pi(a | s)
class PolicyModelPG:
    '''
    A network model that outputs probability p(a|s), for policy gradient methods.
    It is trained to minimize the sum of the returns of the states (maybe subtracted
    from baseline values) each multiplied by the log of the probability of the action.
    '''
    def __init__(self, state_size, hidden_sizes, n_actions, lr=0.01):
        self.policy_net = TorchMultiLayerNetwork(state_size, hidden_sizes, n_actions)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=1)
        self.obs_size = state_size
        self.n_actions = n_actions
        self.hidden_sizes = list(hidden_sizes)
        self.lr = lr

    def partial_fit(self, states, actions, states_vals):
        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions)
        state_vals_v = torch.FloatTensor(states_vals)

        logits_v = self.policy_net(states_v)
        log_prob_v = nn.functional.log_softmax(logits_v, dim=1)
        log_prob_actions_v = state_vals_v * log_prob_v[range(len(states)), actions_v]
        basic_loss_v = -log_prob_actions_v.mean()

        basic_loss_v.backward()
        self.optimizer.step()
        return basic_loss_v.item()

    def predict(self, observation):
        obs_tensor = torch.FloatTensor([observation])
        act_probs_tensor = self.softmax( self.policy_net(obs_tensor) )
        return act_probs_tensor.data.numpy()[0]

    def sample_action(self, obs):
        probs = self.predict(obs)
        num_actions = len(probs)
        return np.random.choice(num_actions, p=probs)

    def best_action(self, obs):
        probs = self.predict(obs)
        return np.argmax(probs)

    def clone(self):
        cp = PolicyModelPG(self.obs_size, self.hidden_sizes, self.n_actions, self.lr)
        policy_state = cp.policy_net.state_dict()
        for k, v in self.policy_net.state_dict().items():
            policy_state[k] = v
            cp.policy_net.load_state_dict(policy_state)
        return cp


# approximates V(s)
class ValueModel:
    def __init__(self, state_size, hidden_sizes, lr=0.001):
        self.value_net = TorchMultiLayerNetwork(state_size, hidden_sizes, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.value_net.parameters(), lr=lr) # inicializado com os parametros da rede

    def partial_fit(self, states, values):
        states_v = torch.FloatTensor(states)
        values_v = torch.FloatTensor(values)
        self.optimizer.zero_grad()  # atencao: o optimizer acessa os parametros de "policy_net" e faz os ajustes nos pesos
        scores_v = self.value_net(states_v)
        loss_v = self.loss_function(scores_v.view(-1), values_v) 
        loss_v.backward()
        self.optimizer.step() 
        return loss_v.item() # converte the Torch para valor escalar

    def predict(self, state):
        state_tensor = torch.FloatTensor([state])
        value_tensor = self.value_net(state_tensor)
        return value_tensor[0,0].item()  # original value: [[V]]


def test_policy(env, policy, deterministic, num_episodes=5, render=False, videorec=None):
    """
    Avalia a política `policy`, usando a melhor ação sempre, de forma determinística.
    - env: o ambiente
    - policy: a política
    - deterministic: `True`, se for usar o método `.best_action(obs)`; `False`, para usar `.sample_action(obs)`
    - num_episodes: quantidade de episódios a serem executados
    - render: defina como True se deseja chamar env.render() a cada passo
    - video: passe uma instância de VideoRecorder (do gym), se desejar gravar
    
    Retorna:
    - um par contendo o valor escalar do retorno médio por episódio e 
       a lista de retornos de todos os episódios
    """
    episodes_returns = []
    total_steps = 0
    for i in range(num_episodes):
        obs = env.reset()
        if render:
            env.render()
        if videorec is not None:
            videorec.capture_frame()
        done = False
        steps = 0
        episodes_returns.append(0.0)
        while not done:
            if deterministic:
                action = policy.best_action(obs)
            else:
                action = policy.sample_action(obs)
            obs, reward, done, _ = env.step(action)
            if render:
                env.render()
            if videorec is not None:
                videorec.capture_frame()
            total_steps += 1
            episodes_returns[-1] += reward
            steps += 1
        print(f"EPISODE {i+1}")
        print("- steps:", steps)
        print("- return:", episodes_returns[-1])
    mean_return = round(np.mean(episodes_returns), 1)
    print("RESULTADO FINAL: média (por episódio):", mean_return, end="")
    print(", episódios:", len(episodes_returns), end="")
    print(", total de passos:", total_steps)
    if videorec is not None:
        videorec.close()
    return mean_return, episodes_returns


class PolicyModelPGWithExploration(PolicyModelPG):
    '''
    Similar to PolicyModelPG, but its loss function has an extra term (entropy of the output)
    to induce more exploration. The exploration factor should be in interval [0.0; 1.0]:
    When it is value is 0.0, the loss function is like PolicyModelPG. A low value is recommended.
    '''
    def __init__(self, state_size, hidden_sizes, n_actions, lr=0.01, exploration_factor=0.1):
        super().__init__(state_size, hidden_sizes, n_actions, lr)
        #self.optimizer.eps = 1e-3
        self.expl_factor = exploration_factor

    def partial_fit(self, states, actions, states_vals):
        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions)
        state_vals_v = torch.FloatTensor(states_vals)

        logits_v = self.policy_net(states_v)
        log_prob_v = nn.functional.log_softmax(logits_v, dim=1)
        log_prob_actions_v = state_vals_v * log_prob_v[range(len(states)), actions_v]
        basic_loss_v = -log_prob_actions_v.mean()

        prob_v = nn.functional.softmax(logits_v, dim=1)
        entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()

        loss_v = basic_loss_v + self.expl_factor * entropy_loss_v 
        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()

    def clone(self):
        cp = PolicyModelPGWithExploration(self.obs_size, self.hidden_sizes, self.n_actions, self.lr, self.expl_factor)
        policy_state = cp.policy_net.state_dict()
        for k, v in self.policy_net.state_dict().items():
            policy_state[k] = v
            cp.policy_net.load_state_dict(policy_state)
        return cp
