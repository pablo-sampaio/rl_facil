import numpy as np
import matplotlib.pyplot as plt

from . import smooth


# future: rever a ORDEM dos parâmetros (e rever onde esta função é usada)
def plot_result(returns, ymax_suggested=None, x_log_scale=False, window=10, return_type='episode', filename=None, cumulative=False):
    '''Exibe um gráfico "episódio/passo x retorno", fazendo a média a cada `window` retornos, para suavizar.
    
    Parâmetros:
    - returns: se return_type=='episode', este parâmetro é uma lista de retornos a cada episódio; se return_type=='step', é uma lista de pares (passo,retorno) 
    - ymax_suggested (opcional): valor máximo de retorno (eixo y), se tiver um valor máximo conhecido previamente
    - x_log_scale: se for True, mostra o eixo x na escala log (para detalhar mais os resultados iniciais)
    - window: permite fazer a média dos últimos resultados, para suavizar o gráfico
    - return_type: use 'episode' ou 'step' para indicar o que representa o eixo x; também afeta como será lido o parâmetro 'returns'
    - filename: se for fornecida uma string, salva um arquivo de imagem ao invés de exibir.
    '''
    plt.figure(figsize=(14,8))

    if return_type == 'episode':
        plt.xlabel('Episódios')
        if cumulative:
            returns = np.array(returns)
            returns = np.cumsum(returns)
        yvalues = smooth(returns, window)
        xvalues = np.arange(1, len(returns)+1)
        plt.plot(xvalues, yvalues)
        plt.title(f"Retorno médio a cada {window} episódios")
    #elif return_type == 'step':
    else:
        print("Attention: 'window' is ignored for 'step' type of returns")
        plt.xlabel('Passos')
        xvalues, yvalues = list(zip(*returns))
        xvalues = np.array(xvalues) + 1
        if cumulative:
            yvalues = np.array(yvalues)
            yvalues = np.cumsum(yvalues)
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


def plot_multiple_results(list_returns, cumulative='no', x_log_scale=False, return_type='episode', window=10, plot_stddev=False, yreference=None):
    '''Exibe um gráfico "episódio/passo x retorno" com vários resultados.
    
    Parâmetros:
    - list_returns: uma lista de triplas (nome do resultado, retorno por episódio/passo, outras informações)
    - cumulative: indica se as recompensas anteriores devem ser acumuladas, para calcular a média histórica por episódio
    - x_log_scale: se for True, mostra o eixo x na escala log (para detalhar mais os resultados iniciais)
    - window: permite fazer a média dos últimos resultados, para suavizar o gráfico; só é usado se cumulative='no'
    - plot_stddev: exibe sombra com o desvio padrão, ou seja, entre média-desvio e média+desvio
    - yreference: if not None, should be an integer, where will be plot a horizontal gray dashed line, used for reference
    '''
    # True and False are here for backward compatibility (remove!)
    if cumulative is None or cumulative is False:
        cumulative = 'no'
    if cumulative is True: 
        cumulative = 'avg'
    assert cumulative in ['no', 'sum', 'avg']
    assert return_type in ['step', 'episode']
    total_steps = list_returns[0][1].shape[1]
    plt.figure(figsize=(14,8))
    for (alg_name, returns) in list_returns:
        xvalues = np.arange(1, total_steps+1)
        if cumulative == 'sum' or cumulative == 'avg':
            # calculate the cumulative sum along axis 1
            cumreturns = np.cumsum(returns, axis=1)
            if cumulative == 'avg':
                cumreturns = cumreturns / xvalues
            yvalues = cumreturns.mean(axis=0)
            std = cumreturns.std(axis=0)
        else:
            yvalues = smooth(returns.mean(axis=0),window)
            std = returns.std(axis=0)
        plt.plot(xvalues, yvalues, label=alg_name)
        if plot_stddev:
            plt.fill_between(xvalues, yvalues-std, yvalues+std, alpha=0.4)
    
    if yreference is not None:
        y_ref_line = [ yreference ] * total_steps
        plt.plot(y_ref_line, linestyle="--", color="gray")

    if x_log_scale:
        plt.xscale('log')
    
    if return_type == 'episode':
        plt.xlabel('Episódio')
        payoff = 'Retorno'
    else:
        plt.xlabel('Passo')
        payoff = 'Recompensa'
    
    plt.ylabel('Retorno')
    
    if cumulative == 'no':
        plt.title(f"{payoff} (média móvel a cada {window})")
    elif cumulative == 'avg':
        gen = payoff[-1]
        plt.title(f"{payoff} acumulad{gen} médi{gen}")
    else:
        gen = payoff[-1]
        plt.title(f"{payoff} acumulad{gen}")
    
    plt.legend()
    plt.show()
