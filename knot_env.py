import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from povray_render.sample_spline import sample_b_spline, sample_equdistance
from dynamics_inference.dynamic_models import physbam_3d
from topology.representation import AbstractState
from topology.state_2_topology import state2topology
from topology.BFS import bfs_all_path
from topology_learning.gen_random_start_states import gen_random_state, gen_random_state_1loop
import datetime
import pdb


class KnotEnv(object):

  def __init__(self, parallel=4):
    self.action_low = np.array([0, -0.5, -0.5, -0.5, -0.5, 0.02])
    self.action_high = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.2])
    self.dynamic_inference = physbam_3d(' -friction 0.13688 -stiffen_linear 0.23208 -stiffen_bending 0.64118 -self_friction 0.46488')
    self.parallel = parallel

  def step(self, traj_param):
    traj_param = np.array(traj_param)
    if self.parallel==1 and traj_param.ndim==1:
        traj_param = traj_param[np.newaxis,:]
    assert(traj_param.shape[0] == self.parallel and traj_param.shape[1] == self.action_low.shape[0])
    # empty dictionary means simulation failed or a trivial topological action
    traj_param = np.clip(traj_param, self.action_low, self.action_high)
    batch_actions = []
    self.traj = [] # for visualization
    for i,tp in enumerate(traj_param):
      action_node = int(tp[0] * 63)
      action_traj = tp[1:-1]
      height = tp[-1]
      knots = [self.start_state[i][action_node][:2]]*3 + [action_traj[0:2]] + [action_traj[2:4]]*3
      traj = sample_b_spline(knots)
      traj = sample_equdistance(traj, None, seg_length=0.01).transpose()
      traj_height = np.arange(traj.shape[0]) * 0.01
      traj_height = np.minimum(traj_height, traj_height[::-1])
      traj_height = np.minimum(traj_height, height)
      traj = np.concatenate([traj, traj_height[:,np.newaxis]], axis=-1)
      self.traj.append(traj)
      moves = traj[1:]-traj[:-1]
      actions = [(action_node, m) for m in moves]
      batch_actions.append(actions)

    state = self.dynamic_inference.execute_batch(self.start_state, batch_actions, return_3d=True)

    reward = [{}]*self.parallel
    done = [True]*self.parallel
    info = [{}]*self.parallel

    for i,st in enumerate(state):
      if st is None:
        state[i] = np.zeros((64,3))
      else:
        start_abstract_state, start_intersections = state2topology(self.start_state[i], update_edges=True, update_faces=True)
        end_abstract_state, end_intersections = state2topology(st, update_edges=True, update_faces=False)
        intersect_points = [i[0] for i in end_intersections] + [i[1] for i in end_intersections]
        if len(set(intersect_points)) == len(intersect_points):
          # the end state does not have more than 2 segments sharing 1 intersection
          paths = bfs_all_path(start_abstract_state, end_abstract_state, max_depth=1)
          if len(paths)==1 and len(paths[0][1])==1:  # there is only one possible topological action
            reward[i] = paths[0][1][0] # the type of topological action
          if len(paths) > 1:  # there are multiple possibilities
            start_intersect = [i[0] for i in start_intersections] + [i[1] for i in start_intersections]
            start_intersect.sort()
            manipulate_idx = np.searchsorted(start_intersect, traj_param[i,0]*63)
            for path, path_action in paths:
                if path_action[0].get('idx') == manipulate_idx or path_action[0].get('over_idx') == manipulate_idx:
                    reward[i] = path_action[0]

    self.end_state = state
    return self.end_state, reward, done, info


  def reset(self):
    #self.start_state = np.zeros((self.parallel, 64,3))
    #self.start_state[:,:,0] = np.linspace(-0.5, 0.5, 64)
    #start_state = np.loadtxt('start_state_1_intersection.txt')
    #self.start_state = np.tile(start_state, (self.parallel,1,1))
    self.start_state = [gen_random_state_1loop() for _ in range(self.parallel)]
    self.start_state = np.array(self.start_state)
    if np.random.rand()>0.5:
        self.start_state[:,:,1]=-self.start_state[:,:,1] # flip
    rotations = np.random.uniform(0,np.pi*2, size=(self.parallel,))
    translations = np.random.uniform(-0.1,0.1,size=(self.parallel,1,2))
    rotations = np.array([[np.cos(rotations), np.sin(rotations)],
                          [-np.sin(rotations), np.cos(rotations)]]).transpose((2,0,1))
    self.start_state[:,:,:2] = np.matmul(self.start_state[:,:,:2], rotations) + translations
    self.start_state = [st for st in self.start_state]
    #self.start_state = [gen_random_state() for _ in range(self.parallel)]
    #self.start_state = [np.concatenate([st, np.zeros((64,1))], axis=1) for st in self.start_state]
    return self.start_state


  def render(self, mode='human', close=False):
    for i in range(self.parallel):
      plt.clf()
      plt.figure()
      plt.plot(self.start_state[i][:, 0], self.start_state[i][:, 1], c='r', label='start')
      plt.plot(self.end_state[i][:, 0], self.end_state[i][:, 1], c='g', label='end')
      plt.plot(self.traj[i][:, 0], self.traj[i][:, 1], c='b', label='traj')
      plt.legend()
      plt.savefig('/home/myyan92/topology_learning/%s-%d.png' % (str(datetime.datetime.now()), i))
      plt.close()
