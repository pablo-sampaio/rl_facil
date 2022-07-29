import time

import numpy as np
import matplotlib.pyplot as plt


def plot_result(returns, ymax_suggested=None, filename=None):
    '''Exibe um gráfico "retornos x recompensas", fazendo a média a cada 100 retornos, para suavizar.     
    Se o parâmetro filename for fornecido, salva o gráfico em arquivo ao invés de exibir.
    
    Parâmetros:
    - returns: lista de retornos a cada episódio
    - ymax_suggested (opcional): valor máximo de retorno (eixo y), se tiver um valor máximo conhecido previamente
    - filename: indique um nome de arquivo, se quiser salvar a imagem do gráfico; senão, o gráfico será apenas exibido
    '''
    # alternative: a moving average
    avg_every100 = [np.mean(returns[i:i+100])
                    for i in range(0, len(returns), 100)]
    xvalues = np.arange(1, len(avg_every100)+1) * 100
    plt.figure(figsize=(14,8))
    plt.plot(xvalues, avg_every100)
    plt.xlabel('Episódios')
    plt.ylabel('Retorno médio')
    if ymax_suggested is not None:
        ymax = np.max([ymax_suggested, np.max(avg_every100)])
        plt.ylim(top=ymax)
    plt.title('Retorno médio a cada 100 episódios')
    if filename is None:
        plt.show()
        print("Nenhum arquivo salvo.")
    else:
        plt.savefig(filename)
        print("Arquivo salvo:", filename)
    plt.close()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_multiple_results(results, cumulative=False, x_log_scale=False):
    '''Exibe um gráfico "retornos x recompensas" para vários resultados.
    
    Parâmetros:
    - results: são triplas (nome do resultado, retorno por episódio, outras informações)
    - cumulative: indica se as recompensas anteriores devem ser acumuladas, para calcular a média histórica por episódio
    - x_log_scale: inficar True se quiser imprimir o eixo x na escala log (que detalha mais os resultados iniciais)
    '''
    total_steps = len(results[0][1])  # no primeiro resultado (#0), pega o tamanho do array de recompensas, que fica na segunda posição (#1)

    if not cumulative:
        # plot all the raw returns, with x linear
        """
        plt.figure(figsize=(14,8))
        for (alg_name, returns, _) in results:
            plt.plot(returns, label=alg_name)
        if x_log_scale:
            plt.xscale('log')
        plt.title(f"Retorno por episódio")
        plt.legend()
        plt.show()"""

        # plot the returns smoothed by a moving average with window 100, with x linear
        plt.figure(figsize=(14,8))
        for (alg_name, returns, _) in results:
            plt.plot(moving_average(returns,50), label=alg_name)
        if x_log_scale:
            plt.xscale('log')
        plt.title("Smoothed 100-reward")
        plt.legend()
        plt.show()

    else:
        # plot cumulative average returns
        plt.figure(figsize=(14,8))
        for (alg_name, returns, _) in results:
            cumulative_average = np.cumsum(returns) / (np.arange(1, total_steps+1))
            plt.plot(cumulative_average, label=alg_name)
        if x_log_scale:
            plt.xscale('log')
        plt.title("Cumulative Average")
        plt.legend()
        plt.show()

    for (alg_name, returns, exec_info) in results:
        print("Summary for " + alg_name)
        print(" - sum rewards (all episodes):", returns.sum())
        #print(" - extra info (algorithm-dependent):")
        #print(exec_info)
        print()

    return
