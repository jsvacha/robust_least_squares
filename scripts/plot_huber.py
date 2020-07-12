#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def huber(x, m):
  if abs(x) < m:
    return x**2
  else:
    return m * (2 * abs(x) - m)

def huberLoss(x, y, m):
  return huber(x, m) + huber(y, m)

if __name__ == "__main__":

  # computation
  m = 1
  win = 10
  increment = 0.1
  X = np.arange(-win, win, increment)
  Y = np.arange(-win, win, increment)
  X, Y = np.meshgrid(X, Y)
  Z = np.zeros(np.shape(X))
  for i in range(np.shape(X)[0]):
    for j in range(np.shape(X)[0]):
      Z[i,j] = huberLoss(X[i,j], Y[i,j], m)

  # Plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  plt.show()
