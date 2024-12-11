import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from cap09.models_torch_pg import TorchMultiLayerNetwork, test_policy


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
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.hidden_sizes = list(hidden_sizes)
        self.lr = lr

    def partial_fit(self, observations, target_acts):
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
    
    def clone(self):
        cp = PolicyModelCrossentropy(self.obs_size, self.hidden_sizes, self.n_actions, self.lr)
        policy_state = cp.policy_net.state_dict()
        cp.policy_net.load_state_dict(policy_state)
        #for k, v in self.policy_net.state_dict().items():
        #    #print(k)
        #    policy_state[k] = v
        return cp

    def save(self, filename):
        torch.save(self.policy_net.state_dict.state_dict(), filename)
    
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
