
from typing import final
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TorchMultiLayerNetwork(nn.Module):
    def __init__(self, input_dim, list_hidden_dims, output_dim, final_activ_fn=None):
        super(TorchMultiLayerNetwork, self).__init__()
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


class PolicyModelCrossentropy:
    ''' 
    A network model that outputs probability p(a|s), trained to minimize 
    crossentropy loss function.
    '''
    def __init__(self, obs_size, hidden_sizes, n_actions, lr=0.001):
        self.policy_net = TorchMultiLayerNetwork(obs_size, hidden_sizes, n_actions)
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)  # recebe os parametros da rede, para poder fazer ajustes neles

    def partial_train(self, observations, target_acts):
        obs_v = torch.FloatTensor(observations)
        acts_v = torch.LongTensor(target_acts)
        self.optimizer.zero_grad()                           # zera os arrays com os gradientes associados aos parâmetros
        action_scores_v = self.policy_net(obs_v)             # faz um "forwards pass" com todas as entradas, calculando automaticamente os gradientes
                                                             # atenção: a "camada" de softmax é ignorada aqui no treinamento -- resultado melhor
        loss_v = self.loss_function(action_scores_v, acts_v) # calcula a cross-entropy entre os valores calculados (para cada ação) pela rede e os valores esperados (uma ação "alvo" específica)
        loss_v.backward()                                    # propaga o erro para as camadas anteriores -- backward pass
        self.optimizer.step()                                # ajusta os pesos
        return loss_v.item()        # converte de Torch para valor escalar

    @torch.no_grad()
    def predict(self, observation):
        obs_tensor = torch.FloatTensor([observation])
        action_probs_tensor = self.softmax( self.policy_net(obs_tensor) )
        return action_probs_tensor.data.numpy()[0] 
    
    def sample_action(self, obs):
        prob_a_s = self.predict(obs)
        return np.random.choice(len(prob_a_s), p=prob_a_s)

    def best_action(self, obs):
        prob_a_s = self.predict(obs)
        return np.argmax(prob_a_s)


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

    def partial_train(self, states, actions, states_vals):
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


# approximates V(s)
class ValueModel:
    def __init__(self, state_size, hidden_sizes, lr=0.001):
        self.value_net = TorchMultiLayerNetwork(state_size, hidden_sizes, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.value_net.parameters(), lr=lr) # inicializado com os parametros da rede

    def partial_train(self, states, values):
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
        return value_tensor.data.numpy()[0][0]  # original value: [[V]]


class PolicyModelPGWithExploration(PolicyModelPG):
    '''
    Similar to PolicyModelPG, but its loss function has an extra term (entropy of the output)
    to induce more exploration. The exploration factor should be in interval [0.0; 1.0]:
    When it is value is 0.0, the loss function is like PolicyModelPG. A low value is recommended.
    '''
    def __init__(self, state_size, hidden_sizes, n_actions, lr=0.01, exploration_factor=0.1):
        super(PolicyModelPGWithExploration, self).__init__(state_size, hidden_sizes, n_actions, lr)
        self.expl_factor = exploration_factor

    def partial_train(self, states, actions, states_vals):
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
        loss_v = (1.0 - self.expl_factor) * basic_loss_v + self.expl_factor * entropy_loss_v 
        print(" - basic loss: ", basic_loss_v.item(), ", entropy loss:", entropy_loss_v.item())

        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()

