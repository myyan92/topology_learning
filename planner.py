import random
import pdb
from state_encoder import *
from topology.representation import AbstractState, reverse_action
from topology.state_2_topology import state2topology
from topology.BFS import generate_next
from topology.reverse_BFS import bfs_all_path as reverse_bfs_all_path
from advanced_buffer import get_reward_key
import gin

@gin.configurable
def get_random_action(obs, reward_key_list):
    topo_state, _ = state2topology(obs, update_edges=True, update_faces=True)
    possible_actions = [ta[1] for ta in generate_next(topo_state)]
    while True:
        intended_action = random.choice(possible_actions)
        unified_action = intended_action.copy()
        if 'left' in unified_action:
            unified_action['left']=1
        if 'sign' in unified_action:
            unified_action['sign']=1
        if get_reward_key(unified_action, obs) in reward_key_list:
            return intended_action

def hash_dict(abstract_action):
    tokens = [k+':'+str(v) for k,v in abstract_action.items()]
    return ' '.join(sorted(tokens))

@gin.configurable
def get_random_action_from_list(obs, reward_key_list, action_list):
    topo_state, _ = state2topology(obs, update_edges=True, update_faces=True)
    possible_actions = [ta[1] for ta in generate_next(topo_state)]
    action_list = [hash_dict(ac) for ac in action_list]
    possible_actions = [ac for ac in possible_actions if hash_dict(ac) in action_list]
    while True:
        intended_action = random.choice(possible_actions)
        unified_action = intended_action.copy()
        if 'left' in unified_action:
            unified_action['left']=1
        if 'sign' in unified_action:
            unified_action['sign']=1
        if get_reward_key(unified_action, obs) in reward_key_list:
            return intended_action

@gin.configurable
def get_fixed_action(obs, reward_key_list, action):
    return action


class goal_planner(object):
    def __init__(self, goal):
        self.goal = goal
        start = AbstractState()
        paths = reverse_bfs_all_path(self.goal, start)
        feasible_states, planned_actions = [], []                
        for reverse_path_state, reverse_path_action in paths:
            path_state = reverse_path_state[::-1]
            path_action = [reverse_action(reverse_path_action[i], reverse_path_state[i], reverse_path_state[i+1])
                           for i in range(len(reverse_path_action))]
            path_action = path_action[::-1]
            for st, ac in zip(path_state, path_action):
                if st in feasible_states:
                    planned_actions[feasible_states.index(st)].append(ac)
                else:
                    feasible_states.append(st)
                    planned_actions.append([ac])
        self.feasible_states, self.planned_actions = feasible_states, planned_actions

    def is_feasible(self, state):
        return state in self.feasible_states

    def not_feasible(self,state):
        return not self.is_feasible(state)

    def reached_goal(self, state):
        return state==self.goal

    def get_action(self, geo_state, model_keys):
        state, _ = state2topology(geo_state, True, True)
        if self.is_feasible(state):
            actions = self.planned_actions[self.feasible_states.index(state)]
            return random.choice(actions)


def encode(obs, intended_actions):
    encode_dicts = [buffer_2_model(*state_2_buffer(ob, ac)) for ob,ac in
                    zip(obs, intended_actions)]
    encode_dicts = zip(*encode_dicts)
    obs, over, under = encode_dicts
    obs = np.array(obs)
    over_seg_dict = pad_batch(over)
    under_seg_dict = pad_batch(under)
    return obs, over_seg_dict, under_seg_dict
