# 1. Ideias de Projetos

Os projetos serão individuais.

## Implementar Programação Dinâmica - Bruno

Implementar algum desses algoritmos: policy iteration ou o value iteration.

Usar o ambiente Frozen-Lake, com um wrapper para embutir o modelo do ambiente. Usar uma representação de estados (x,y), para facilitar.


## Epsilon Decrescente - Daniel

Fazer um ou mais algoritmos usando o $\epsilon$ decrescente. Criar um parâmetro para esse decaimento.

Otimizar o algoritmo para todos os parâmetros.

Fazer uma *análise de sensibilidade* da taxa de decaimento.


## Learning Rate Decrescente - Diego

Semelhante ao anterior.


## Aplicar em Outros Ambientes

Adotar ambientes mais desafiadores e tentar achar uma boa solução. Existem muitos ambientes de *jogos* (Atari, Minecraft, Mario, Futebol, etc) e ambientes de *robótica*.

Existem opções no gym e a internet está cheia de opções alternativas que seguem a mesma interface do gym. Veja esta lista:
https://www.gymlibrary.ml/environments/third_party_environments/


## AWS Deep Racer

É um caso especial do de cima. 

Mas você vai treinar especificamente para o ambiente de competição simulado criado pela Amazon:
https://aws.amazon.com/pt/deepracer/


## Google Research Footbal

Ambiente de futebol do Google.

https://github.com/google-research/football


## Projetos Baseados em Artigos da Comunidade

Não estou falando de artigos acadêmicos, mas de artigos de experts e "curiosos" da comunidade.

Existem diversos artigos interessantes que você pode tentar reproduzir e melhorar os resultados. 
Ou apenas mudar os experimentos para mudar outros fatores.

Exemplos:
- https://towardsdatascience.com/using-deep-q-learning-in-fifa-18-to-perfect-the-art-of-free-kicks-f2e4e979ee66


## Refazer experimentos do livro

Escolher alguns dos experimentos registrados no livro para reproduzir ou alguns dos exercícios. Algumas possibilidades:
- Example 6.6: implementar o ambiente e refazer o experimento, mostrando a política e plotando o gráfico
- Exercício 5.12

## Usar Softmax - Mateus Wei

Nos algoritmos baseados em Q-table (Q-Learning, Expected-SARSA), ao invés de epsilon-greedy, usar uma política softmax e comparar os resultados.


## Implementar outros algoritmos do livro e comparar

Exemplos:
- Algoritmo MonteCarlo *off-policy*: ver seção 5.7
- Double Q-Learning: seção 6.7
- Algoritmos TD de n passos *off-policy*: ver seções 7.3 e 7.5
- Dyna-Q: seção 8.2

## Algoritmos Evolucionários - João Victor

Aplicar em algum ambiente e fazer experimentos cuidadosos.


# 2. Ideias de Seminários

Para apresentar de forma individual ou em dupla.

## Programação Dinâmica - Bruno

Explicar o assunto.

## Contextual Bandits - J. Rodrigues, Lucas Lins

Caso dos bandits em que existe um estado, que afeta nas probabilidades dos ações (das "alavancas").

Tenho sugestão de um cap. de livro para esse problema.


## Apresentar uma Plataforma Cloud para RL - Mateus Lins (GCP)

Apresentar as funcionalidades de Cloud para RL de alguma plataforma (GCP, AWS, Azure ou IBM) para uso comercial, etc.

Escolher uma e detalhar as funcionalidades, dando exemplos de aplicações.

Se possível, mostrar exemplo prático de como usar.


## Aprendizagem por Reforço Multiagente - Daniel, Mateus Wei e Diego

Existem muitas técnicas, mas eu tenho sugestões de dois caps. de diferentes livros. Falar de ambos.


## Dyna-Q - Fábio

Falar dos algoritmos desta família. Capítulo 8 do livro de Sutton e Barto, até seção 8.3 ou 8.4.


## Algoritmos MonteCarlo Off-policy - Maely e Matheus Felipe

Pode ser feito junto com um projeto no mesmo tema. Ver seção 5.7 e seções próximas.


## Algoritmos TD de n passos off-policy - Giulia e Luiz Fernando

Pode ser feito junto com um projeto no mesmo tema. Ver seção 7.3 e seções próximas.


## Extensões do DQN - Hugo (Rainbow)

Extensões do DQN.

Pode escolher uma dessas opções para falar: Rainbow ou APE-X.


## SAC

Algoritmo da família Policy Gradient, para ações contínuas. Estado da arte.


## Abordagem Teacher-Student

Uma alternativa ao uso de um currículo estático. 


## Algoritmos Evolucionários para RL - Joao Victor e Sávio

Algoritmos evolucionários usados para treinamento de RL.


# Datas dos Seminários 

19/09
- Maely e Matheus Felipe - MonteCarlo Off-policy 
- Giulia e Luiz Fernando - TD de n passos off-policy 
- Fábio - Dyna-Q

22/set
- Hugo - Rainbow
- Mateus Lins - Serviços de RL da GCP
- Bruno - Progr. Dinâmica

26/set
- Rodrigues e Lucas - Contextual Bandits (antecipar para 22)
- Daniel, Mateus Wei e Diego - Aprendizagem por Reforço Multiagente
- Joao Victor - Algoritmos Evolucionários

29/set
- Sávio - Algoritmos Evolucionários
- Pablo - AlphaZero

