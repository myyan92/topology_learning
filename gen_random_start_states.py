import numpy as np
import random
import glob
import gin
from physbam_python.state_to_mesh import state_to_mesh
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

@gin.configurable
def gen_random_state():
  # using a OU process
  mu=0.1 # correspond to average curvature
  while np.abs(mu)>0.07:
    mu = np.random.normal(0.0,0.04)
  decay=0.95 # amount of correlation
  sigma=0.1 # amount of added Gaussian

  edge_theta = [0.0]
  for i in range(62):
      new_theta = decay * edge_theta[-1] + sigma * np.random.normal()
      edge_theta.append(new_theta)
  edge_theta = np.array(edge_theta) + mu*np.arange(63)

  edge_length = 1.0/63.0
  # integrating tangent to rope state
  state = [np.array([0.0,0.0])]
  for i in range(63):
      new_point = state[-1] + edge_length * np.array([np.cos(edge_theta[i]), np.sin(edge_theta[i])])
      state.append(new_point)

  state= np.array(state)
  state = state - np.mean(state, axis=0)

  # random translation, rotation and scaling
  translation = np.random.uniform(-0.1,0.1,size=2)
  rotation = np.random.uniform(-3.14,3.14)
  scaling = np.random.normal(1.0,0.1)
  matrix = np.array([[np.cos(rotation), np.sin(rotation)],[-np.sin(rotation), np.cos(rotation)]])
  state = np.dot(state, matrix) * scaling + translation
  state = np.concatenate([state, np.zeros((64,1))], axis=-1)
  state = state_to_mesh(state)
  state = state.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
  return state

@gin.configurable
def load_random_state(file_pattern):
  files = glob.glob(file_pattern)
  file = random.choice(files)
  state = np.loadtxt(file)
  return state

if __name__ == "__main__":
  state = gen_random_state()
  plt.plot(state[:,0],state[:,1])
  plt.xlim([-0.5,0.5])
  plt.ylim([-0.5,0.5])
  plt.show()
