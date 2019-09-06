from advanced_buffer import Buffer
import numpy as np
import pdb

reward_keys = ["move-R1_left-1_sign-1",
               "move-R1_left-1_sign--1",
               "move-R1_left--1_sign-1",
               "move-R1_left--1_sign--1",
               "move-R2_left-1_over_before_under-1",
               "move-R2_left-1_over_before_under--1",
               "move-R2_left--1_over_before_under-1",
               "move-R2_left--1_over_before_under--1",
               "move-R2_left-1_diff",
               "move-R2_left--1_diff",
               "move-cross_endpoint-over_sign-1",
               "move-cross_endpoint-under_sign-1",
               "move-cross_endpoint-over_sign--1",
               "move-cross_endpoint-under_sign--1",
              ]

buffers = [Buffer(key) for key in reward_keys]

start_state = np.loadtxt('../topology/start_state_1_intersection.txt')
action = np.random.rand(6)

for idx in range(3):
    for left in [-1,1]:
        for sign in [-1,1]:
            reward = {'move':'R1', 'idx':idx, 'left':left, 'sign':sign}
            for buffer in buffers:
                buffer.put(start_state, action, reward)
for idx in range(3):
    for left in [-1,1]:
        for over_before_under in [-1,1]:
            reward = {'move':'R2', 'over_idx':idx, 'under_idx':idx, 'left':left, 'over_before_under':over_before_under}
            for buffer in buffers:
                buffer.put(start_state, action, reward)
for over_idx in range(3):
    for under_idx in range(3):
        if over_idx != under_idx:
            for left in [-1,1]:
                reward = {'move':'R2', 'over_idx':over_idx, 'under_idx':under_idx, 'left':left, 'over_before_under':1}
                for buffer in buffers:
                    buffer.put(start_state, action, reward)
for over_idx in range(3):
    for under_idx in range(3):
        if over_idx != under_idx:
            for sign in [-1,1]:
                reward = {'move':'cross', 'over_idx':over_idx, 'under_idx':under_idx, 'sign':sign}
                for buffer in buffers:
                    buffer.put(start_state, action, reward)

for buffer in buffers:
    obs, action, _, over, under = buffer.get(10)

