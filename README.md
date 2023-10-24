# Aprendizagem por Reforço (RL) Fácil

Repositório criado com propósitos didáticos, contendo códigos de exemplo de Reinforcement Learning (RL) em Python com explicações em português.

Abaixo, damos uma explicação do conteúdo das pastas, que estão organizadas como se fossem capítulos.

# Cap. 1 - Problema dos Multi-Armed Bandits (Caça-Níqueis)

Aqui, vocês encontram códigos que implementam versões do "Multi-armed Bandit Problem" de maneira parecida com os ambientes "gym"
e também implementações de alguns algoritmos.

Ambientes:
- `SimpleMultiarmedBandit` - recompensas têm valor 0 ou 1, com certa probabilidade distinta por ação
- `GaussianMultiarmedBandit` - recompensas têm valores contínuos quaisquer seguindo uma distribuição Normal (Gaussiana), com média distinta por ação

Soluções:
- **Random** e **Greedy** (gulosa) - soluções "ruins", para compartação
- **Epsilon-greedy** - faz uma ação qualquer com probabilidade dada pelo parâmetro epsilon ($\epsilon$) ou escolhe (de forma gulosa) a ação de melhor média Q
- **UCB** - escolhe ação com base em uma fórmula que combina: (i) a melhor média de recompensa da ação (Q); e (ii) um termo que valoriza ações pouco executadas (para explorar)


# Cap. 2 - Ambientes

Nesta pasta, vocês encontram códigos que mostram como executar alguns **ambientes `gymnasium`** e como tentar resolver manualmente um deles.

Veja também como criar *wrappers* de ambientes.


# Cap. 3 - Processos de Decisão de Markov (MDPs)

Alguns códigos Python e notebooks Jupyter explicando MDPs e ilustrando conceitos sobre eles usando o `gymnasium``, onde
alguns ambientes funcionam precisamente como um MDP.

Ilustramos esses conceitos básicos:
- **Trajetória** - são os detalhes de um episódio
- **Retorno** (não-descontado e descontado) - é a soma das recompensas, que pode ter os valores futuros atenuados progressivamente
- **Política** - recebe um estado e decide a próxima ação a ser executada

Com base nisso, introduzimos os conceitos de **funções de valor** ($V$ e $Q$) associadas a uma política.

Mostramos *Algoritmos de Monte Carlo (para previsão) para estimar V e Q*. Ele são a base para os próximos algoritmos.


# Cap. 4 - Algoritmos Monte Carlo (para Controle)

Implementações de algoritmos de aprendizagem por reforço que são baseados em estimativas da função $Q$ (valor estado-ação)
representadas na forma de tabela (array bidimensional ou similar), que costuma ser chamada *Q-Table*.

Foi implementado o algoritmo:
- *Monte-Carlo Control* (duas versões) - gera episódios inteiros, para atualizar Q (método on-policy)


# Cap. 5 - Métodos TD-Learning Básicos

Aqui tem um notebook explicando as equações de Bellman, usadas nos algoritmos de TD-Learning. Aqui são dadas vários 
implementações de algoritmos TD-Learning. Todos estão implementados em *Q-Table*, como os anteriores.

Algoritmos implementados:
- **Q-Learning** - atualiza a cada passo, roda uma polítiva epsilon-greedy, mas atualiza como greedy (off-policy)
- **SARSA** - atualiza a cada passo, roda uma polítiva epsilon-greedy e atualiza coerentemente (on-policy)
- **Expected-SARSA** - como os anteriores, mas pode ser on-policy ou off-policy (mas está implementado de forma on-policy)

# Cap. 6 - SARSA de n Passos / Técnicas Auxiliares

Vemos o **SARSA de n passos**, que atualiza a *Q-Table* com base nos dados coletadas nos últimos $n$ passos,
onde este valor é um parâmetro adicional.

Taambém vemos técnicas que auxiliam nos algoritmos anteriores (e em alguns algoritmos futuros):
- Como lidar com ambientes contínuos
- Como otimizar os (muitos) parâmetros dos algoritmos e da discretização


# Cap. 7 - Algoritmos com Modelo (Dyna-Q) e Algoritmos de MDP de Recompensa Média (Opcional)

Nesta parte do curso, vemos uma nova formulação de MDP especialmente apropriada para *tarefas continuadas*, ou seja, tarefas que não 
têm um estado terminal. Nestes MDPs, o objetivo é achar a política que maximize a recompensa média (a cada passo).

Existem algoritmos específicos propostos com base nesta formulação.

Algoritmo implementado:
- **Differential Q-Learning**: versão do Q-Learning para tarefas continuadas, baseado na formulação de recompensa média


# Cap. 8 - DQN

Aqui, veremos o DQN, sucessor do Q-Learning que usa uma rede neural para substituir a *Q-Table*.

Em especial, esse algoritmo pode ser aplicado naturalmente em ambientes com estados contínuos e até em jogos de Atari
(ou outros ambientes com observações dadas na forma de imagens).


# Cap. 9 - Métodos Policy Gradient - Versão Monte Carlo

Nesta parte, vemos métodos que aprendem uma política de forma explícita. Em especial, vamos representá-la com alguma rede neural.
Em especial, focamos nos métodos da família mais importante do momento, que é chamada *policy gradient*. 

Estes métodos usam funções de custo (*loss function*) específicas para RL.

Algoritmos:
- *REINFORCE* - é o método mais básico, que é uma técnica Monte Carlo (roda episódigos completos, para calcular os $G_t$)
- *REINFORCE-Adv* - melhoria do anterior, que usa uma rede neural separada para aprender o $V(s)$

# Cap. 10 - Métodos Actor-Critic

Aqui, vemos os métodos *policy gradient* combinados com TD-Learning, que são os métodos chamados **actor-critic**.

Algoritmos (nomenclatura minha):

- *VAC-1* - método ator crítico mais básico de 1 passos, que é uma versão TD-Learning do REINFORCE
- *VAC-N* - método ator crítico básico de n passos, que é uma extensão do anterior


# Cap. 11 - Bibliotecas

Neste ponto do curso, vamos ver algumas bibliotecas que oferecem algoritmos do estado da arte.


# CapExtra - Outros

Aqui, mostramos outro algoritmo baseado em política que não é da família *policy gradient*. Trata-se do algoritmo *Cross-Entropy*,
que é uma aplicação do método de otimização *cross-entropy* (entropia cruzada) ao problema da aprendizagem por reforço.
De certa forma, ele transforma um problema de RL em um problema de *classificação* da aprendizagem supervisionada.

Aqui, complementando o cap. 6, vemos mais alguns algoritmos que aprendem a política diretamente, sendo esta representada 
como uma rede neural (ou outro modelo diferenciável). Porém, estes métodos usam funções de custo (*loss function*) específicas
para RL.
