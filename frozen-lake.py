import gym
import numpy as np
import pandas as pd

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
)
'''
Implementation of tabular Q - Learning in the frozen lake environment.
'''
env = gym.make("FrozenLakeNotSlippery-v0")

#env = gym.make('FrozenLake-v0')
# 0 1 2 3  up down right left
env.reset()
env.render()
np.random.seed(8565)
np.set_printoptions(suppress=True)
a_size = env.action_space.n
s_size = env.observation_space.n

episodes = 882
test_episodes = 100
max_steps = 50

lr = 0.8
g = .95

#controls the rate at which the agent explores or exploits

E = .5

Q = np.zeros((s_size, a_size))     # Q initial values
count = 0

for episode in range(episodes):
    s = env.reset()
    done = False

    if episode/episodes >=.7:
        E = .1
    for step in range(max_steps):
        if (np.random.uniform(0,1) < E):
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s,:])
        s_ , r, d, _ = env.step(a) # take a
        Q[s,a] = Q[s,a] + lr*(r + g*np.max(Q[s_,:]) - Q[s,a])
        s = s_
        if d:
            break
        s = s_


#test

print(Q)
print("failed tries: "+str(count))
total_reward = 0
for episode in range(test_episodes):
    s = env.reset()
    step = 0
    done = False
    for step in range(max_steps):
        #env.render()

        a = np.argmax(Q[s,:])
        #print(a)
        s_, reward, done, info = env.step(a)
        total_reward += reward

        if done:
            break
        s = s_
print("Total reward across {} episodes : {}".format(test_episodes,total_reward))

env.close()
