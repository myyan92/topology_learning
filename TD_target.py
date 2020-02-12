import numpy as np
from advanced_buffer import get_reward_key
from planner import get_random_action, get_fixed_action, encode
from topology.state_2_topology import state2topology
from state_encoder import unifying_transform_encode
import gin
import pdb

def hash_dict(abstract_action):
    tokens = [k+':'+str(v) for k,v in abstract_action.items()]
    return ' '.join(sorted(tokens))

@gin.configurable
class TDTarget(object):
    """
    We use this class to calculate state values.

    """
    def __init__(self, models,
                 topo_action_func, planner_not_feasible_func, planner_reached_goal_func):
        self.model_dict = {model.scope:model for model in models}
        self.topo_action_func = topo_action_func
        self.planner_not_feasible = planner_not_feasible_func
        self.planner_reached_goal = planner_reached_goal_func

    def run(self, sess, obs):
        state_values = np.zeros((len(obs),))
        original_index, next_trans_obs, next_trans_intended_actions = [], [], []
        # filter not feasible states and fill in 0.
        # filter reached_goal and fill in 1.
        # batching to evaluate next state's state value
        for idx,ob in enumerate(obs):
            topo, intersections = state2topology(ob, True, True)
            intersections = [it[0] for it in intersections] + [it[1] for it in intersections]
            intersections.sort()
            intersections = np.array([0]+[it+1 for it in intersections]+[64])
            if np.amin(intersections[1:]-intersections[:-1])<2:
                state_values[idx] = 0.0
                continue
            if self.planner_not_feasible(topo):
                state_values[idx] = 0.0
            elif self.planner_reached_goal(topo):
                state_values[idx] = 1.0
            else:
                next_intended_action = self.topo_action_func(ob, self.model_dict.keys())
                obs_u, _, ia_u, _ = unifying_transform_encode(ob, None, next_intended_action)
                next_trans_obs.append(obs_u)
                next_trans_intended_actions.append(ia_u)
                original_index.append(idx)

        next_reward_keys = [get_reward_key(ia_u, obs_u) for ia_u, obs_u in zip(next_trans_intended_actions, next_trans_obs)]
        # group state and fetch model
        model_keys = set(next_reward_keys)
        for key in model_keys:
            model = self.model_dict[key]
            sublist_trans_obs = [ob_u for ob_u, k in zip(next_trans_obs, next_reward_keys) if k==key]
            sublist_trans_ia = [ia_u for ia_u, k in zip(next_trans_intended_actions, next_reward_keys) if k==key]
            model_inputs = encode(sublist_trans_obs, sublist_trans_ia)
            sublist_state_values = model.predict_batch_vf(sess, *model_inputs)
            # fill in
            idx = 0
            for i,k in enumerate(next_reward_keys):
                if k==key:
                    state_values[original_index[i]]=sublist_state_values[idx]
                    idx += 1
        return state_values
