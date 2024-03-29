import random
import numpy as np
import functools
from tree import Tree
import pdb
from povray_render.sample_spline import sample_b_spline, sample_equdistance
from dynamics_inference.dynamic_models import physbam_3d
from topology.state_2_topology import state2topology
from model_GRU_attention_3 import Model as Model_Actor
from model_GRU_C3 import Model as Model_Critic
import heuristic as heuristic

class RRT(object):
    def __init__(self, x_init, topology_path, action_sampling_funcs, max_samples, parallel_sim):
        """
        Template RRT planner
        :param x_init: tuple, initial location
        :param topology_path: a list of topological states from BFS. Including init state.
        :param action_sampling_funcs: a list of functions with length len(topology_path)-1,
               each function takes geometric state and predict robot action param.
        :param max_samples: int, maximum number of samples
        """
        self.samples_taken = 0
        self.max_samples = max_samples
        self.parallel = parallel_sim
        self.x_init = x_init
        self.topology_path = topology_path
        self.action_sampling_funcs = action_sampling_funcs
        self.tree = Tree()
        self.mental_dynamics = physbam_3d(physbam_args=' -friction 0.13688 -stiffen_linear 0.23208 -stiffen_bending 0.64118 -self_friction 0.46488') # -gravity 0.0 -convergence_tol 1e-3')

    def select_node(self, tree):
        prob = np.array(tree.V_priority) * 2
        prob = np.exp(prob)
        prob = prob / np.sum(prob)
        idx = np.random.choice(len(tree.V_priority), p=prob)
        return (tree.V_priority[idx], idx, tree.V_data[idx])

    def sample_branch(self, parent):
        parent_state = np.array(parent).reshape((128,3))
        parent_obs = 0.5*(parent_state[:64]+parent_state[64:])
        parent_topology, _ = state2topology(parent_obs, update_edges=True, update_faces=False)
        parent_index = self.topology_path.index(parent_topology)
        action_sampler = self.action_sampling_funcs[parent_index]
        traj_param = action_sampler(parent_obs)
        traj_param = np.clip(traj_param, np.array([0.0, -0.5, -0.5, -0.5, -0.5, 0.02]),
                                         np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.2]))
        action_node = int(traj_param[0]*63)
        action_traj = traj_param[1:-1]
        height = traj_param[-1]
        knots = [parent[action_node][:2]]*3 + [action_traj[0:2]] + [action_traj[2:4]]*3
        traj = sample_b_spline(knots)
        traj = sample_equdistance(traj, None, seg_length=0.01).transpose()
        traj_height = np.arange(traj.shape[0]) * 0.01
        traj_height = np.minimum(traj_height, traj_height[::-1])
        traj_height = np.minimum(traj_height, height)
        traj = np.concatenate([traj, traj_height[:,np.newaxis]], axis=-1)
        moves = traj[1:]-traj[:-1]
        actions = [(action_node,m) for m in moves]

#        action_node = 0
#        move = np.random.normal(size=3)
#        move = move / np.linalg.norm(move) * 0.05
#        actions = [(action_node, move)]
        return actions

    def score_priority(self, parent_priority, parent, child):
        if child is None:
            return None
        parent_topology, intersections = state2topology(parent, update_edges=True, update_faces=False)
        child_topology, intersections = state2topology(child, update_edges=True, update_faces=False)
        parent_index = self.topology_path.index(parent_topology)
        try:
            child_index = self.topology_path.index(child_topology)
        except ValueError:
            return None
        if child_index < parent_index:
            return None
        intersections = [it[0] for it in intersections] + [it[1] for it in intersections]
        intersections = np.array([0] + sorted(intersections) + [64])
        seglength = intersections[1:]-intersections[:-1]
#        if np.amin(seglength) == 0:
#            return None # not a promissing state for next action
        print("Achieving ", child_index)
#        np.savetxt('%d.txt'%(self.samples_taken), child)
        child_priority = parent_priority * (2**(child_index-parent_index))
        return child_priority

    def is_solution(self, node):
        topology, _ = state2topology(node, update_edges=True, update_faces=False)
        if topology == self.topology_path[-1]:
            return True
        else:
            return False

    def rrt_search(self):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """
        self.tree.add_root((1.0, 0, self.x_init))

        while self.samples_taken < self.max_samples:
            # expand in parallel
            print("new batch")
            parents = [self.select_node(self.tree) for _ in range(self.parallel)]
            trajs = [self.sample_branch(parent[2]) for parent in parents]
            parent_states = [parent[2] for parent in parents]
            child_states = self.mental_dynamics.execute_batch(parent_states, trajs, return_traj=False, reset_spring=(self.samples_taken==0))
            parent_obs = [0.5*(ps[:64]+ps[64:]) if ps is not None else None for ps in parent_states]
            child_obs = [0.5*(ps[:64]+ps[64:]) if ps is not None else None for ps in child_states]
            priorities = [self.score_priority(parent_node[0], parent, child) for parent_node, parent, child in zip(parents, parent_obs, child_obs)]
            for priority, parent, child_st, child_ob, traj in zip(priorities, parents, child_states, child_obs, trajs):
                if priority is not None:
                    self.samples_taken += 1
                    self.tree.add_leaf(parent, (priority, self.samples_taken, child_st), traj)
                    if self.is_solution(child_ob):
                        waypoints, actions = self.tree.reconstruct_path((priority, self.samples_taken, child_st))
                        return waypoints, actions
        print("Cannot find solution!")
        return None


import sys, pickle
from copy import deepcopy
import tensorflow as tf
from model_GRU_attention import Model
from topology.representation import AbstractState
from topology.BFS import bfs
from physbam_python.state_to_mesh import state_to_mesh
from action_sampler import *


if __name__ == "__main__":
    sampling_mode = sys.argv[1]

    init_state= np.zeros((64,3))
    init_state[:,0] = np.linspace(-0.5,0.5,64)
#    init_state[:,2] = 0.2
    init_state = state_to_mesh(init_state)
    init_state = init_state.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))

    init_topology = AbstractState()
    topology_path = [deepcopy(init_topology)]
    init_topology.Reide1(0, left=1, sign=1)
    topology_path.append(deepcopy(init_topology))
    init_topology.cross(0,1, sign=1)
    topology_path.append(deepcopy(init_topology))
    init_topology.cross(2,0, sign=1)
    topology_path.append(deepcopy(init_topology))

    if sampling_mode == 'random':
        action_sampling_funcs = [random_uniform_sampler(), random_uniform_sampler(), random_uniform_sampler()]
    elif sampling_mode == 'heuristic':
#        action_sampling_funcs = [random_gaussian_heuristic_sampler([0.9, 0.0, 0.2, -0.2, -0.2, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
#                                 random_gaussian_heuristic_sampler([0.05, -0.2, 0.0, -0.05, -0.1, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05, 0.03]),
#                                 random_gaussian_heuristic_sampler([0.6, -0.05, -0.1, -0.08, -0.15, 0.03],[0.05, 0.03, 0.03, 0.03, 0.03, 0.03])] # TO BE ADJUSTED
        action_sampling_funcs = [random_gaussian_heuristic_sampler([0.9, 0.0, 0.2, -0.2, -0.2, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05])]
        def interpolate_gaussian_heuristic_sampler(obs, anno_states, anno_actions):
            heuristic_mean = heuristic.interpolate_heuristic(anno_states, anno_actions, obs[:,:2])
            action_std = np.array([0.03,0.05,0.05,0.05,0.05,0.05])
            return np.random.normal(loc=heuristic_mean, scale=action_std)

        folder = '1loop_states/annotate_0on1'
        anno_states, anno_actions = heuristic.load_heuristic(folder)
        action_sampling_funcs.append(functools.partial(interpolate_gaussian_heuristic_sampler, anno_states=anno_states, anno_actions=anno_actions))

        folder = '2intersect_states/annotate_2on0'
        anno_states, anno_actions = heuristic.load_heuristic(folder)
        action_sampling_funcs.append(functools.partial(interpolate_gaussian_heuristic_sampler, anno_states=anno_states, anno_actions=anno_actions))

    elif sampling_mode == 'model':
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=16,
            intra_op_parallelism_threads=16)
        tf_config.gpu_options.allow_growth=True
        sess = tf.Session(config=tf_config)

        def load_change_scope(model, load_path, new_scope, old_scope):
            vars_to_load = model.get_trainable_variables()
            reader = tf.train.load_checkpoint(load_path)
            for var in vars_to_load:
                new_name = var.name
                old_name = new_name.replace(new_scope, old_scope)
                if old_name.endswith(':0'):
                    old_name = old_name[:-2]
                tensor = reader.get_tensor(old_name)
                sess.run(tf.assign(var, tensor))

        intended_action = {'move':'R1', 'idx':0, 'left':1, 'sign':1}
        actor_model = Model_Actor('move-R1_left-1_sign-1_actor')
        actor_model.build()
        load_change_scope(actor_model, '0to1-moves-randstate-m3/models/model-move-R1_left-1_sign-1-630',
                          'move-R1_left-1_sign-1_actor', 'move-R1_left-1_sign-1')
        critic_model = Model_Critic('move-R1_left-1_sign-1_critic')
        critic_model.build()
        load_change_scope(critic_model, '0to1-moves-randstate_mC3/models/model-move-R1_left-1_sign-1-3400',
                          'move-R1_left-1_sign-1_critic', 'move-R1_left-1_sign-1')
        action_sampling_funcs = [model_sampler(sess, actor_model, critic_model, intended_action)]

        intended_action = {'move':'cross', 'over_idx':0, 'under_idx':1, 'sign':1}
        actor_model = Model_Actor('move-cross_endpoint-over_sign-1_actor')
        actor_model.build()
        load_change_scope(actor_model, '1to2-cross-endpointover-sign1-randstate-m3/models/model-move-cross_endpoint-over_sign-1-6700',
                          'move-cross_endpoint-over_sign-1_actor', 'move-cross_endpoint-over_sign-1')
        critic_model = Model_Critic('move-cross_endpoint-over_sign-1_critic')
        critic_model.build()
        load_change_scope(critic_model, '1to2-cross-endpointover-sign1-randstate_mC3_0on1/models/model-move-cross_endpoint-over_sign-1-44450',
                          'move-cross_endpoint-over_sign-1_critic', 'move-cross_endpoint-over_sign-1')
        action_sampling_funcs.append(model_sampler(sess, actor_model, critic_model, intended_action))

        intended_action = {'move':'cross', 'over_idx':2, 'under_idx':0, 'sign':1}
#        actor_model = Model_Actor('move-cross_endpoint-under_sign-1_actor')
#        actor_model.build()
#        load_change_scope(actor_model, '2to3-cross-endpointunder-sign1-randstate-m3/models/model-move-cross_endpoint-under_sign-1-1600')
#                          'move-cross_endpoint-under_sign-1_actor', 'move-cross_endpoint-under_sign-1')
        critic_model = Model_Critic('move-cross_endpoint-under_sign-1_critic')
        critic_model.build()
        load_change_scope(critic_model, '2to3-cross-endpointunder-sign1-randstate_mC3_2on0/models/model-move-cross_endpoint-under_sign-1-108000',
                          'move-cross_endpoint-under_sign-1_critic', 'move-cross_endpoint-under_sign-1')
        action_sampling_funcs.append(model_sampler(sess, None, critic_model, intended_action))

    rrt_search = RRT(init_state, topology_path, max_samples = 2000000,
                     action_sampling_funcs = action_sampling_funcs,
                     parallel_sim=1)
    trajectory = rrt_search.rrt_search()
    if trajectory is not None:
        print('found a solution after %d samples.' % (rrt_search.samples_taken))
        waypoints, actions = trajectory
        np.save('rrt_waypoints.npy', np.array(waypoints))
        for i,ac in enumerate(actions):
            np.save('rrt_actions_%d.npy'%(i), np.array(ac))


