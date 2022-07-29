# Aprendizagem por Reforço (RL) Fácil

Repositório criado com propósitos didáticos, contendo códigos de exemplo de Reinforcement Learning (RL) em Python com explicações em português.

Abaixo, damos uma explicação do conteúdo das pastas, que estão organizadas como se fossem capítulos.

# Cap. 1 - Ambientes
Nesta pasta, vocês encontram códigos que mostram como executar e como tentar resolver manualmente alguns ambientes do "gym".

Vejam o vídeo da aula 2.

# Cap. 2 - Problema dos Multi-Armed Bandits (Caça-Níqueis)
Aqui, vocês encontram códigos que implementam versões do "Multi-armed Bandit Problem" de maneira parecida com os ambientes "gym"
e também implementações de alguns algoritmos.

Ambientes:
- SimpleMultiarmedBandit - recompensas têm valor 0 ou 1, com certa probabilidade distinta por ação
- GaussianMultiarmedBandit - recompensas têm valores contínuos quaisquer seguindo uma distribuição Normal (Gaussiana), com média distinta por ação

Soluções:
- *Random* e *Greedy* (gulosa) - soluções ruins, para compartação
- *Epsilon-greedy* - faz uma ação qualquer com probabilidade dada pelo parâmetro epsilon ($\epsilon$) ou escolhe (de forma gulosa) a ação de melhor média Q
- *UCB* - escolha ação de forma "gulosa" quanto ao valor de uma fórmula que combina:
  - a melhor média de recompensa da ação (Q), 
  - com um termo que valoriza ações pouco executadas (para explorar)

# Cap. 3 - Processos de Decisão de Markov (MDPs)
Alguns códigos Python e notebooks Jupyter explicando MDPs e ilustrando conceitos sobre eles usando o gym.
Basicamente, cada ambiente do "gym" pode ser visto como um MDP (ou quase isso).

Em especial, ilustramos esses conceitos:
- Trajetória - são os detalhes de um episódio
- Retorno (descontado) - é a soma das recompensas, que pode ter os valores futuros atenuados progressivamente
- Política - recebe um estado e decide a próxima ação a ser executada

Com base nisso, introduzimos os conceitos de funções de valor ($V$ e $Q$) associadas a uma política, em um MDP/ambiente.

# Cap. 4 - Métodos Baseados em Q-Table

Implementações de algoritmos de aprendizagem por reforço que são baseados em estimativas da função $Q$ (valor estado-ação)
representadas na forma de tabela (array bidimensional ou similar), que costuma ser chamada *Q-Table8.

Foram implementados os algoritmos:
- *Monte-Carlo Control* - gera episódios inteiros, para atualizar Q (método on-policy, com leve toque offline)
- *Q-Learning* - atualiza a cada passo, roda uma polítiva epsilon-greedy, mas atualiza como greedy (off-policy, online)
- *Expected-SARSA*  - atualiza a cada passo roda uma polítiva epsilon-greedy e atualiza coerentemente (on-policy, online)


# Cap. 5 - Técnicas Auxiliares

Vemos como implementar técnicas que auxiliam nos algoritmos anteriores (e em alguns algoritmos futuros):
- Como lidar com ambientes contínuos
- Como otimizar os (muitos) parâmetros dos algoritmos e da discretização

Também estão aqui:
- Um notebook explicando as equações de Bellman
- Uma extensão dos algoritmos vistos antes, para fazer atualizações usando dados de "n passos" 

# Cap. 6 - Método Cross-Entropy

Nesta parte, vemos um método que aprende uma política diretamente, onde a política é representada por uma rede neural.

Basicamente, ele transforma um problema de RL em um problema de classificação da aprendizagem supervisionada.

