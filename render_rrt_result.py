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
#    action = [np.insert(ac[1], 3, ac[0]/63.0) for ac in action]
#    actions.append(np.array(action))
    actions.append(list(action))


physbam_args = ' -friction 0.13688 -stiffen_linear 0.23208 -stiffen_bending 0.64118 -self_friction 0.46488' # -rope_width 0.005'
dynamics = physbam_3d(physbam_args)

state = waypoints[0]

#state = state_to_mesh(state)
#state = state.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
state_trajs = []
for i,action in enumerate(actions):
    state_traj = dynamics.execute(state, action, return_traj=True, reset_spring=True)
#    state_traj = rollout_single(state, action, physbam_args=' -dt 1e-3 ' + physbam_args,
#                                return_3d=True, return_traj=True, keep_files=True, input_raw=i>0, return_raw=True,
#                                curve_proto_file=os.path.join(root_dir, 'sim_%d_raw.input'%(i)),
#                                action_proto_file=os.path.join(root_dir, 'sim_%d_raw.action'%(i)),
#                                output_dir=os.path.join(root_dir, 'sim_%d_raw'%(i)))
    state_trajs.extend(state_traj)
    state = state_traj[-1]

