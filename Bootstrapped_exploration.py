import numpy as np
import pandas as pd 
import time


class Env(object):

    def __init__(self):
        self.reset()

    def step(self, action):
        pass

    def reset(self):


np.random.seed(2) #reproducible

STATES_NUM = 6 # the length of the 1 dimensional world
ACTIONS = ['left','right']
EPSLION = 0.9
ALPHA = 0.1 # learning rate
LAMDA = 0.9 # discount factor
MAX_EPISODES = 13 # episodes
FRESH_TIME = 0.01

def build_q_table(STATES_NUM, actions):
    table = pd.DataFrame(np.zeros((STATES_NUM, len(actions))), columns = actions,)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSLION or (state_actions.all() == 0)):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

# input: state & action. output: next state & reward
def get_env_feedback(S,A):
    if A == 'right':
        if S == STATES_NUM - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S 
        else:
            S_ = S - 1
    return S_, R 

def update_env(S, episode, step_counter):
    env_list = ['-']*(STATES_NUM-1) + ['T']
    if S == 'terminal':
        interaction = 'episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                      ', end = '')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(STATES_NUM, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S,q_table)
            #print(A)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + LAMDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.ix[S,A] += ALPHA * (q_target - q_predict)
            S = S_
            print(S)

            update_env(S,episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nq_table:\n')
    print(q_table)
