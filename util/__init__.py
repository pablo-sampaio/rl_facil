
import numpy as np

def smooth(data, window):
  data = np.array(data)
  n = len(data)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i-window+1)
    y[i] = data[start:(i+1)].mean()
  return y

#def moving_average(x, w):
#    return np.convolve(x, np.ones(w), 'valid') / w
