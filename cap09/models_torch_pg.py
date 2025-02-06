import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchMultiLayerNetwork(nn.Module):
    def __init__(self, input_dim, list_hidden_dims, output_dim, final_activ_fn=None):
        super().__init__()
        layers = []
        last_dim = input_dim
        for dim in list_hidden_dims:
            layers.append(nn.Linear(last_dim, dim, bias=True))
            layers.append(nn.ReLU())
            last_dim = dim
        layers.append(nn.Linear(last_dim, output_dim, bias=True))
        if final_activ_fn is not None:
            layers.append(final_activ_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PolicyModelPG:
    '''
    A network model that outputs probability p(a|s), for policy gradient methods.
    It is trained to minimize the sum of the returns of the states (maybe subtracted
    from baseline values) each multiplied by the log of the probability of the action.
    '''
    def __init__(self, state_size, hidden_sizes, n_actions, lr=0.01, device=DEFAULT_DEVICE):
        self.policy_net = TorchMultiLayerNetwork(state_size, hidden_sizes, n_actions).to(device)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=1)
        self.obs_size = state_size
        self.n_actions = n_actions
        self.hidden_sizes = list(hidden_sizes)
        self.lr = lr
        self.device = device

    def update_weights(self, states, actions, states_vals):
        states = np.asarray(states)
        actions = np.asarray(actions)
        states_vals = np.asarray(states_vals)
        
        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(states).to(self.device)
        actions_v = torch.LongTensor(actions).to(self.device)
        state_vals_v = torch.FloatTensor(states_vals).to(self.device)

        logits_v = self.policy_net(states_v)
        log_prob_v = nn.functional.log_softmax(logits_v, dim=1)
        log_prob_actions_v = state_vals_v * log_prob_v[range(len(states)), actions_v]
        basic_loss_v = -log_prob_actions_v.mean()

        basic_loss_v.backward()
        self.optimizer.step()
        return basic_loss_v.item()

    # rever esse nome
    def predict(self, observation):
        with torch.no_grad():
            #obs_tensor = torch.FloatTensor([observation]).to(self.device)
            reshaped_obs = np.asarray(observation).reshape(1, -1)
            obs_tensor = torch.FloatTensor(reshaped_obs).to(self.device)
            act_probs_tensor = self.softmax(self.policy_net(obs_tensor))
        return act_probs_tensor.cpu().data.numpy()[0]

    def sample_action(self, obs):
        probs = self.predict(obs)
        num_actions = len(probs)
        return np.random.choice(num_actions, p=probs)

    def best_action(self, obs):
        probs = self.predict(obs)
        return np.argmax(probs)

    def clone(self):
        cp = PolicyModelPG(self.obs_size, self.hidden_sizes, self.n_actions, self.lr, self.device)
        cp.policy_net.load_state_dict(self.policy_net.state_dict())
        return cp


# approximates V(s)
class ValueModel:
    def __init__(self, state_size, hidden_sizes, lr=0.001, device=DEFAULT_DEVICE):
        self.value_net = TorchMultiLayerNetwork(state_size, hidden_sizes, 1).to(device)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.value_net.parameters(), lr=lr)
        self.obs_size = state_size
        self.hidden_sizes = list(hidden_sizes)
        self.lr = lr
        self.device = device

    def update_weights(self, states, target_values):
        states = np.asarray(states)
        target_values = np.asarray(target_values)

        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(states).to(self.device)
        values_v = torch.FloatTensor(target_values).to(self.device)
        scores_v = self.value_net(states_v)
        loss_v = self.loss_function(scores_v.view(-1), values_v)
        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()  # converte para valor escalar

    def predict(self, state):
        with torch.no_grad():
            #state_tensor = torch.FloatTensor([state]).to(self.device)
            reshaped_state = np.asarray(state).reshape(1, -1)
            state_tensor = torch.FloatTensor(reshaped_state).to(self.device)
            value_tensor = self.value_net(state_tensor)
        return value_tensor[0, 0].item()  # original value: [[V]]

    def predict_batch(self, state_list):
        with torch.no_grad():
            states = np.asarray(state_list)
            states_tensor = torch.FloatTensor(states).to(self.device)
            values_tensor = self.value_net(states_tensor)
        # converts from tensor to numpy
        return values_tensor.cpu().data.numpy().flatten()

    def clone(self):
        cp = ValueModel(self.obs_size, self.hidden_sizes, self.lr, self.device)
        cp.value_net.load_state_dict(self.value_net.state_dict())
        return cp


def test_policy(env, policy, deterministic, num_episodes=5):
    """
    Avalia a política `policy`, usando a melhor ação sempre, de forma determinística.
    - env: o ambiente
    - policy: a política
    - deterministic: `True`, se for usar o método `.best_action(obs)`; `False`, para usar `.sample_action(obs)`
    - num_episodes: quantidade de episódios a serem executados
    
    Retorna:
    - um par contendo o valor escalar do retorno médio por episódio e 
       a lista de retornos de todos os episódios
    """
    episodes_returns = []
    total_steps = 0
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        episodes_returns.append(0.0)
        while not done:
            if deterministic:
                action = policy.best_action(obs)
            else:
                action = policy.sample_action(obs)
            obs, reward, termi, trunc, _ = env.step(action)
            done = termi or trunc
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
    return mean_return, episodes_returns


class PolicyModelPGWithExploration(PolicyModelPG):
    '''
    Similar to PolicyModelPG, but its loss function has an extra term (entropy of the output)
    to induce more exploration. The exploration factor should be in interval [0.0; 1.0]:
    When it is value is 0.0, the loss function is like PolicyModelPG. A low value is recommended.
    '''
    def __init__(self, state_size, hidden_sizes, n_actions, lr=0.01, exploration_factor=0.1, device=DEFAULT_DEVICE):
        super().__init__(state_size, hidden_sizes, n_actions, lr, device)
        #self.optimizer.eps = 1e-4
        self.expl_factor = exploration_factor

    def update_weights(self, states, actions, states_vals):
        states = np.asarray(states)
        actions = np.asarray(actions)
        states_vals = np.asarray(states_vals)

        self.optimizer.zero_grad()
        states_v = torch.FloatTensor(states).to(self.device)
        actions_v = torch.LongTensor(actions).to(self.device)
        states_vals_v = torch.FloatTensor(states_vals).to(self.device)

        logits_v = self.policy_net(states_v)
        log_prob_v = nn.functional.log_softmax(logits_v, dim=1)
        log_prob_actions_v = states_vals_v * log_prob_v[range(len(states)), actions_v]
        basic_loss_v = -log_prob_actions_v.mean()

        prob_v = nn.functional.softmax(logits_v, dim=1)
        entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()

        loss_v = basic_loss_v + self.expl_factor * entropy_loss_v
        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()

    def clone(self):
        cp = PolicyModelPGWithExploration(self.obs_size, self.hidden_sizes, self.n_actions, self.lr, self.expl_factor, self.device)
        '''policy_state = cp.policy_net.state_dict()
        for k, v in self.policy_net.state_dict().items():
            policy_state[k] = v
        cp.policy_net.load_state_dict(policy_state)
        '''
        cp.policy_net.load_state_dict(self.policy_net.state_dict())
        return cp
