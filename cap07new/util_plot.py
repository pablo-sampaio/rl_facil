import numpy as np
import matplotlib.pyplot as plt


def smooth(data, window):
  data = np.array(data)
  n = len(data)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i-window+1)
    y[i] = data[start:(i+1)].mean()
  return y


def plot_result(returns, ymax_suggested=None, window=100, filename=None):
    '''Exibe um gráfico "retornos x recompensas", fazendo a média a cada 100 retornos, para suavizar.     
    Se o parâmetro filename for fornecido, salva o gráfico em arquivo ao invés de exibir.
    
    Parâmetros:
    - returns: lista de retornos a cada episódio
    - ymax_suggested (opcional): valor máximo de retorno (eixo y), se tiver um valor máximo conhecido previamente
    - filename: indique um nome de arquivo, se quiser salvar a imagem do gráfico; senão, o gráfico será apenas exibido
    '''
    plt.figure(figsize=(14,8))
    smoothed_returns = smooth(returns, window)
    xvalues = np.arange(1, len(returns)+1)
    plt.plot(xvalues, smoothed_returns)
    plt.xlabel('Episódios')
    plt.ylabel('Retorno')
    if ymax_suggested is not None:
        ymax = np.max([ymax_suggested, np.max(smoothed_returns)])
        plt.ylim(top=ymax)
    plt.title(f"Retorno médio a cada {window} episódios")
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        print("Arquivo salvo:", filename)
    plt.close()


#def moving_average(x, w):
#    return np.convolve(x, np.ones(w), 'valid') / w

def plot_multiple_results(results, cumulative=False, x_log_scale=False, window=100):
    '''Exibe um gráfico "episódios x retornos" com vários resultados.
    
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
            plt.plot(smooth(returns,window), label=alg_name)
            #plt.plot(moving_average(returns,100), label=alg_name)
        if x_log_scale:
            plt.xscale('log')
        plt.xlabel('Episódios')
        plt.ylabel('Retorno')
        plt.title(f"Retorno médio a cada {window} episódios")
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
        plt.xlabel('Episódios')
        plt.ylabel('Retorno')
        plt.title("Retorno acumulado médio")
        plt.legend()
        plt.show()

    for (alg_name, returns, exec_info) in results:
        print("Summary for " + alg_name)
        print(" - sum rewards (all episodes):", returns.sum())
        #print(" - extra info (algorithm-dependent):")
        #print(exec_info)
        print()

    return
