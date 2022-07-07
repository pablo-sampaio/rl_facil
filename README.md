# Aprendizagem por Reforço (RL) Fácil

Repositório criado com propósitos didáticos, contendo códigos de exemplo de Reinforcement Learning (RL) em Python com explicações em português.

# Pasta "1_ambientes"
Nesta pasta, vocês encontram códigos que mostram como executar e como tentar resolver manualmente alguns ambientes do "gym".

Vejam o vídeo da aula 2.

# Pasta "2_ma_bandits"
Aqui, vocês encontram códigos que implementam versões do "Multi-armed Bandit Problem" de maneira parecida com os ambientes "gym"
e também implementa alguns algoritmos.

Ambientes:
- SimpleMultiarmedBandit - recompensas têm valor 0 ou 1, com certa probabilidade distinta por ação
- GaussianMultiarmedBandit - recompensas têm valores contínuos quaisquer seguindo uma distribuição Normal (Gaussiana), com média distinta por ação

Soluções:
- *Random* e *Greedy* (gulosa) - soluções ruins, para compartação
- *Epsilon-greedy* - faz uma ação qualquer com probabilidade dada pelo parâmetro epsilon ($\epsilon$) ou escolhe (de forma gulosa) a ação de melhor média Q
- *UCB* - escolha ação de forma "gulosa" quanto ao valor de uma fórmula que combina:
  - a melhor média de recompensa da ação (Q), 
  - com um termo que valoriza ações pouco executadas (para explorar)


