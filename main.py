import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from agent import myEnv as Agent
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

np.random.seed(1)


def build_q_nnet(n_vars, n_actions, alpha):
    model = Sequential()
    model.add(Dense(10, input_shape=(n_vars,),
                    bias_initializer="RandomNormal",
                    activation="relu"))
    model.add(Dense(n_actions))
    model.compile(loss='mse', optimizer=Adam(learning_rate=alpha))
    return model


def choose_action(state, nnet, epsilon):
    state_actions = nnet.predict(state.reshape(1, -1))[0]
    # act non-greedy or state-action have no value
    if np.random.uniform() < epsilon or (state_actions == 0).all():
        act = np.random.randint(len(state_actions))
    else:   # act greedy
        act = np.argmax(state_actions)
    return act


def rl(env, epsilon, alpha, gamma, n_episodes,
       sleep_time=0.01, actions=8, vis=True, max_steps=1000,
       decay=0.9, min_epsilon=0.01):

    state = env._get_state()
    q_nnet = build_q_nnet(len(state), actions, alpha)
    steps_hist = []
    rewards_hist = []

    for episode in range(n_episodes):
        S = env.reset()
        done = False
        total_reward = 0

        for step in range(max_steps):
            act = choose_action(S, q_nnet, epsilon)
            # take action & get next state and reward
            S_, R, done, _ = env.step(act)
            total_reward += R

            if not done:
                # next state is not terminal
                q_target = R + gamma*q_nnet.predict(S_.reshape(1, -1)).max()
            else:
                q_target = R            # next state is terminal
                print('Episodio %s: total steps = %s, total reward = %.2f' % (
                    episode + 1, step + 1, total_reward))

            target_vector = q_nnet.predict(S.reshape(1, -1))
            target_vector[0][act] = q_target
            q_nnet.fit(S.reshape(1, -1), target_vector,
                       verbose=0, epochs=1)  # update

            S = S_  # move to next state

            if vis and not done:
                env.render()
                time.sleep(sleep_time)

            if done:
                steps_hist.append(step + 1)
                rewards_hist.append(step + 1)
                break
            elif step + 1 == max_steps:
                print(f"Episodio {episode + 1}: no alcanza tras {step + 1} pasos")

        if epsilon > min_epsilon:
            epsilon *= decay

    return q_nnet, episode + 1, steps_hist, rewards_hist


def main():
    env = Agent(mode='easy')
    q_nnet, _ = rl(
        env,
        epsilon=0.1,
        alpha=0.01,
        gamma=1.0,
        n_episodes=10,
        vis=True,
        decay=0.9
        max_steps=1000,
    )


if __name__ == '__main__':
    main()
