from numpy.core.fromnumeric import reshape
from viewer import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import pandas as pd
np.random.seed(1)


def build_q_nnet(n_vars, n_actions, alpha):
    model = Sequential()
    model.add(Dense(5,input_shape=(n_vars,),
                   bias_initializer="RandomNormal",
                   activation="relu"))
    model.add(Dense(n_actions))
    model.compile(loss='mse',optimizer=Adam(lr=alpha))
    return model


def choose_action(state, nnet, epsilon):
    # This is how to choose an action
    # devuelve 0 o 1: 0 si left, 1 si right

    state_actions = nnet.predict(state.reshape(1,-1))[0]
    if (np.random.uniform() < epsilon) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        act = np.random.randint(len(state_actions))
    else:   # act greedy
        act = np.argmax(state_actions)
    return act


def rl(env, epsilon, alpha, gamma, n_episodes,
       sleep_time, actions, vis=True):
    
    state = env._get_state()
    q_nnet = build_q_nnet(len(state), actions, alpha)
    
    for episode in range(n_episodes):
        step_counter = 0
        S = env.reset()
        done = False

        while not done:
            act = choose_action(S, q_nnet, epsilon)
            S_, R, done, _ = env.step(act)  # take action & get next state and reward
            
            if not done:
                q_target = R + gamma*q_nnet.predict(S_.reshape(1,-1)).max() # next state is not terminal
            else:
                q_target = R            # next state is terminal
                interaction = 'Episodio %s: total steps = %s' % (episode+1, step_counter)
                print('\r{}'.format(interaction), end='')

            target_vector = q_nnet.predict(S.reshape(1,-1))
            target_vector[0][act] = q_target
            q_nnet.fit(S.reshape(1,-1), target_vector, verbose=0, epochs=1) # update
            
            S = S_  # move to next state
            
            if vis and not done:
                env.render()
                time.sleep(sleep_time)
            step_counter += 1
        
    return q_nnet

env = myEnv()

q_nnet = rl(env, 
            epsilon = 0.4, 
            alpha = 0.01,
            gamma = 0.9,
            n_episodes = 13,
            sleep_time = 0.01, 
            actions = 8,
            vis = True)


# while True:
#     env.reset()
#     env.render()
#     accion = int(input("acción (0 a 8): "))
#     nsteps = int(input("nsteps (>0):    "))
#     for i in range(nsteps):
#         env.step(accion)
#         env.render()
