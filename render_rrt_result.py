from physbam_python.rollout_physbam_3d import rollout_single
from dynamics_inference.dynamic_models import physbam_3d
from physbam_python.state_to_mesh import state_to_mesh
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys, os

root_dir = sys.argv[1]
# open loop actions
actions = []
waypoints = np.load(os.path.join(root_dir, 'rrt_waypoints.npy'))

num_trajs = waypoints.shape[0]-1
for i in range(num_trajs):
    action = np.load(os.path.join(root_dir, 'rrt_actions_%d.npy'%(i)), allow_pickle=True)
    actions.append(list(action))


physbam_args = ' -friction 0.13688 -stiffen_linear 0.23208 -stiffen_bending 0.64118 -self_friction 0.46488' # -gravity 0.0 -convergence_tol 1e-3' # -rope_width 0.005'
dynamics = physbam_3d(physbam_args)

state = waypoints[0]
#state = state_to_mesh(state)
#state = state.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
state_trajs = []

for i,action in enumerate(actions):
    state_traj = dynamics.execute(state, action, return_traj=True, reset_spring=True)
    state_trajs.extend(state_traj)
    state = state_traj[-1]

