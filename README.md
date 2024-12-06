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
- **Random** e **Greedy** (gulosa) - soluções "ruins", para comparação
- **Epsilon-greedy** - faz uma ação qualquer com probabilidade dada pelo parâmetro epsilon ($\epsilon$) ou escolhe a melhor ação conhecida
- **UCB** - escolhe ação com base em uma fórmula que combina a média de recompensa da ação e um termo que valoriza ações pouco executadas


# Cap. 2 - Ambientes

Nesta pasta, vocês encontram códigos que mostram como executar alguns **ambientes `gymnasium`** e como tentar resolver manualmente um deles.

Veja também como criar *wrappers* de ambientes.


# Cap. 3 - Processos de Decisão de Markov (MDPs)

Alguns códigos Python e notebooks Jupyter explicando MDPs e ilustrando conceitos sobre eles usando o `gymnasium`, onde
alguns ambientes funcionam precisamente como um MDP.

Ilustramos esses conceitos básicos:
- **Trajetória** - são os detalhes de um episódio
- **Retorno** (não-descontado e descontado) - é a soma das recompensas, que pode ter os valores futuros atenuados progressivamente
- **Política** - recebe um estado e decide a próxima ação a ser executada

Com base nisso, introduzimos os conceitos de **funções de valor** ($V$ e $Q$) associadas a uma política.

Mostramos *Algoritmos Monte Carlo (para previsão) para estimar V e Q*. Ele são a base para os próximos algoritmos.


# Cap. 4 - Algoritmos Monte Carlo (para Controle)

Implementações de algoritmos de aprendizagem por reforço que são baseados em estimativas da função $Q$ (valor estado-ação)
representadas na forma de uma tabela (array bidimensional ou similar) que costuma ser chamada *Q-Table*.

Algoritmo apresentado e implementado:
- **Algoritmo Monte-Carlo para Controle** (duas versões) - gera episódios inteiros, para atualizar Q (método on-policy) e 
aprender uma política "ótima"


# Cap. 5 - Métodos TD-Learning Básicos

Aqui tem um notebook explicando as equações de Bellman, usadas nos algoritmos de TD-Learning. Aqui são dadas vários 
implementações de algoritmos TD-Learning. Todos fazem atualizações (aprendem) *a cada passo* e são implementados 
com *Q-Tables*, como os anteriores.

Algoritmos implementados:
- **Q-Learning** - roda uma política epsilon-greedy, mas atualiza como greedy (off-policy)
- **SARSA** - roda uma política epsilon-greedy e atualiza coerentemente (on-policy)
- **Expected-SARSA** - generaliza os dois acima (mas está implementado on-policy)


# Cap. 6 - SARSA de n Passos / Técnicas Auxiliares

Vemos o algoritmo **SARSA de n passos**, que atualiza a *Q-Table* com base nos dados coletadas nos últimos $n$ passos,
onde este valor é um parâmetro adicional.

Taambém vemos técnicas que auxiliam nos algoritmos anteriores (e em alguns algoritmos futuros):
- Como lidar com ambientes contínuos
- Como otimizar os (muitos) parâmetros dos algoritmos e da discretização


# Cap. 7 - Algoritmos com Modelo

Nesta parte do curso, vemos *algoritmos com modelo*. Estes algoritmos usam alguma informação sobre o funcionamento 
(a *dinâmica*) do ambiente. 

Em especial, apresentamos o **Dyna-Q**, que é uma extensão simples do *Q-Learning* que
- ao interagir com o ambiente, atualiza *Q* (*RL direta*)
- e também aprende um modelo do ambiente
- e usa este modelo para aprender sem interações com o ambiente (*RL indireta*)


# Cap. 8 - DQN

Aqui, veremos o DQN, sucessor do Q-Learning que usa uma rede neural para substituir a *Q-Table*.

Em especial, esse algoritmo pode ser aplicado naturalmente em ambientes com estados contínuos e até em jogos de Atari
(ou outros ambientes com observações dadas na forma de imagens).


# Cap. 9 - Métodos Policy Gradient - Versão Monte Carlo

Nesta parte, vemos métodos que aprendem uma política de forma explícita - métodos *baseados em política*. 
Em especial, vamos representar a política com alguma rede neural e vamos focar na família chamada *policy gradient*. 

Estes métodos usam funções de custo (*loss function*) específicas para RL. Todos os algoritmos vistos aqui são
técnicas de Monte Carlo, que rodam episódios inteiros antes de atualizar.

Algoritmos:
- **REINFORCE** - é o método mais básico
- **REINFORCE com baseline** - é uma melhoria simples do anterior, que usa o *retorno médio* como baseline
- **REINFORCE com advantage** - caso especial do anterior que usa como uma rede neural adicional para representar $V(s)$


# Cap. 10 - Métodos Actor-Critic

Aqui, vemos os métodos *policy gradient* combinados com TD-Learning, que são os métodos chamados **actor-critic**.

Algoritmos (nomenclatura minha):

- **VAC-1** - algoritmo ator crítico mais básico, de 1 passo
- **VAC-N** - algoritmo ator crítico básico de *n* passos, extensão do anterior


# Cap. 11 - Bibliotecas

Neste ponto do curso, vamos ver algumas bibliotecas que oferecem algoritmos do estado da arte.


# Cap. 12 - Formulação de Recompensa Média

Neste capítulo, vamos focar em **tarefas continuadas**, ou seja, tarefas que não têm um estado terminal. Vamos ver, 
em especial, os *MDPs de recompensa média*, que são uma formulação alternativa do MDP (Markov Decision Process) 
específica para esse tipo de tarefa. 

Nos MDPs de recompensa média, o objetivo é achar a política que maximize a recompensa média (por passo). Existem algoritmos específicos propostos com base nesta formulação. 

Aqui, nós explicamos e implementamos, o algoritmo **Differential Q-Learning**, que é uma adaptação do Q-Learning para esta
nova formulação.


# CapExtra - Outros

Aqui, mostramos outro algoritmo *baseado em política* (mas que não é da família *policy gradient*). Trata-se do algoritmo *Cross-Entropy de Lapan*, encontrado no livro "Deep Reinforcement Learning Hands-On" (Maxim Lapan). Ele aproxima o problema
de controle da aprendizagem por reforço de um problema de *classificação* da aprendizagem supervisionada!

Também mostramos um exemplo de uso de a biblioteca **ray rllib**, que oferece uma grande coleção de algoritmos de 
Aprendizagem por Reforço.


# Agradecimentos e Referências

Este projeto, no geral, é fruto do meu esforço de concentrar algoritmos com simplicidade de implementação, em uma ordem
adequada para aprender progressivamente o assunto. 

Porém, nem todo o código foi desenvolvido exclusivamente por mim. 
Baseei-me em diversas fontes, incluindo livros, cursos online, artigos da web e exemplos fornecidos por bibliotecas 
(como o stable-baselines).
Os códigos aproveitados de outras fontes possuem comentários específicos indicando sua origem (ou inspiração).

Pretendo listar aqui todas as referências específicas para os repositórios utilizados.

É importante observar que este projeto está disponível gratuitamente, sem custos. Se houver alguma preocupação com o uso 
de segmentos de código específicos, sinta-se à vontade para entrar em contato, e eu tratarei da questão prontamente.


# Acknowledgments and References

*This project, overall, is the result of my effort to concentrate algorithms with simplicity of implementation in a suitable order to progressively learn the subject.*

*However, not all the code was developed exclusively by me. I relied on various sources, including books, online courses, web articles, and examples provided by libraries (such as stable-baselines). Codes borrowed from other sources have specific comments indicating their origin (or inspiration).*

*I intend to list here all the specific references for the repositories used.*

*It is important to note that this project is available for free, without cost. If there are any concerns about the use of specific code segments, feel free to contact me, and I will address the issue promptly.*
