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


# future: rever a ORDEM dos parâmetros (e rever onde esta função é usada)
def plot_result(returns, ymax_suggested=None, x_log_scale=False, window=10, return_type='episode', filename=None):
    '''Exibe um gráfico "episódio/passo x retorno", fazendo a média a cada `window` retornos, para suavizar.
    
    Parâmetros:
    - returns: se return_type=='episode', este parâmetro é uma lista de retornos a cada episódio; se return_type=='step', é uma lista de pares (passo,retorno) 
    - ymax_suggested (opcional): valor máximo de retorno (eixo y), se tiver um valor máximo conhecido previamente
    - x_log_scale: se for True, mostra o eixo x na escala log (para detalhar mais os resultados iniciais)
    - window: permite fazer a média dos últimos resultados, para suavizar o gráfico
    - return_type: use 'episode' ou 'step' para indicar o que representa o eixo x; também afeta como será lido o parâmetro 'returns'
    - filename: se for fornecida uma string, salva o gráfico em arquivo ao invés de exibir.
    '''
    plt.figure(figsize=(14,8))

    if return_type == 'episode':
        plt.xlabel('Episódios')
        yvalues = smooth(returns, window)
        xvalues = np.arange(1, len(returns)+1)
        plt.plot(xvalues, yvalues)
        plt.title(f"Retorno médio a cada {window} episódios")
    #elif return_type == 'step':
    else:
        plt.xlabel('Passos')
        xvalues, yvalues = list(zip(*returns))
        xvalues = np.array(xvalues) + 1
        plt.plot(xvalues, yvalues)
        plt.title(f"Retorno médio a cada {window} passos")

    if x_log_scale:
        plt.xscale('log')

    plt.ylabel('Retorno')
    if ymax_suggested is not None:
        ymax = np.max([ymax_suggested, np.max(yvalues)])
        plt.ylim(top=ymax)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        print("Arquivo salvo:", filename)
    
    plt.close()


#def moving_average(x, w):
#    return np.convolve(x, np.ones(w), 'valid') / w

def plot_multiple_results(list_returns, cumulative=False, x_log_scale=False, return_type='episode', window=10, plot_stddev=False):
    '''Exibe um gráfico "episódio/passo x retorno" com vários resultados.
    
    Parâmetros:
    - list_returns: uma lista de triplas (nome do resultado, retorno por episódio/passo, outras informações)
    - cumulative: indica se as recompensas anteriores devem ser acumuladas, para calcular a média histórica por episódio
    - x_log_scale: se for True, mostra o eixo x na escala log (para detalhar mais os resultados iniciais)
    - window: permite fazer a média dos últimos resultados, para suavizar o gráfico; só é usado se cumulative=False
    '''
    if not cumulative:
        # plot the returns smoothed by a moving average with the given window
        plt.figure(figsize=(14,8))
        for (alg_name, returns, std) in list_returns:
            xvalues = np.arange(1, len(returns)+1)
            plt.plot(xvalues, smooth(returns,window), label=alg_name)
            if plot_stddev:
                plt.fill_between(xvalues, returns-std, returns+std, alpha=0.4)
        
        if x_log_scale:
            plt.xscale('log')
        
        if return_type == 'episode':
            plt.xlabel('Episódio')
        else:
            plt.xlabel('Passo')
        
        plt.ylabel('Retorno')
        plt.title(f"Retorno médio a cada {window} episódios")
        plt.legend()
        plt.show()

    else:
        # plot cumulative average returns
        plt.figure(figsize=(14,8))
        for (alg_name, returns, _) in list_returns:
            xvalues = np.arange(1, len(returns)+1)
            cumulative_average = np.cumsum(returns) / xvalues
            plt.plot(xvalues, cumulative_average, label=alg_name)
        
        if x_log_scale:
            plt.xscale('log')
        
        plt.xlabel('Episódio' if return_type=='episode' else 'Passo')
        plt.ylabel('Retorno')
        plt.title("Retorno acumulado médio")
        plt.legend()
        plt.show()

    return
