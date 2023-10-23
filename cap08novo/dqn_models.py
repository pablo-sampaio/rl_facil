import torch
import torch.nn as nn

import numpy as np


class DQNNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class MLP(nn.Module):
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
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


def test_Qpolicy(env, Qpolicy, epsilon=0.0, num_episodes=5, render=False, videorec=None):
    """
    Avalia a política `Qpolicy` escolhendo de forma epsilon-greedy.
    - env: o ambiente
    - Qpolicy: um rede que representa a função Q(s,a) para ser usada como política epsilon-greedy
    - epsilon: probabilidade de ser feita uma escolha aleatória das ações
    - num_episodes: quantidade de episódios a serem executados
    - render: defina como True se deseja chamar env.render() a cada passo
    - video: passe uma instância de VideoRecorder (do gym), se desejar gravar
    Retorna:
    - um par contendo o valor escalar do retorno médio por episódio e 
       a lista de retornos de todos os episódios
    """
    episodes_returns = []
    total_steps = 0
    num_actions = env.action_space.n
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
            if epsilon > 0.0 and np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                state_a = np.array([obs], copy=False)
                state_v = torch.tensor(state_a)
                q_vals_v = Qpolicy(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())
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
