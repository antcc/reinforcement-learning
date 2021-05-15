import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import load_model
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
import time
import numpy as np
from agent import myEnv


def build_q_nnet(n_vars, n_actions, alpha, lr_decay):
    model = Sequential()
    model.add(Dense(50, input_shape=(n_vars,),
                    bias_initializer="RandomNormal",
                    activation="relu"))
    model.add(Dense(20, bias_initializer="RandomNormal",
                    activation="relu"))
    model.add(Dense(n_actions))

    if lr_decay:
        lr = ExponentialDecay(
            alpha,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
    else:
        lr = alpha

    model.compile(loss='mse', optimizer=SGD(learning_rate=lr))
    return model


def replay(q_nnet, gamma, mem, n_actions, batch_size=100):
    if batch_size <= 0:
        return q_nnet

    n_samples = min(batch_size, len(mem))
    minibatch = random.sample(list(mem), n_samples)
    X_train = np.zeros((n_samples, minibatch[0][0].shape[0]))
    y_train = np.zeros((n_samples, n_actions))

    for i, (S, A, R, S_, done) in enumerate(minibatch):
        if not done:
            # next state is not terminal:
            q_target = R + gamma*q_nnet.predict(S_.reshape(1, -1)).max()
        else:
            q_target = R  # next state is terminal

        target_vector = q_nnet.predict(S.reshape(1, -1))
        target_vector[0][A] = q_target
        X_train[i] = S
        y_train[i] = target_vector[0]

    q_nnet.train_on_batch(X_train, y_train)   # update
    return q_nnet


def choose_action(state, nnet, epsilon):
    state_actions = nnet.predict(state.reshape(1, -1))[0]
    # act non-greedy or state-action have no value
    if np.random.uniform() < epsilon or (state_actions == 0).all():
        act = np.random.randint(len(state_actions))
    else:   # act greedy
        act = np.argmax(state_actions)
    return act


def test_policy(env, q_nnet, vis=True,
                sleep_time=0.001,
                max_steps=1000):
    S = env.reset()
    total_reward = 0

    for step in range(max_steps):
        act = np.argmax(q_nnet.predict(S.reshape(1, -1))[0])
        S_, R, done, _ = env.step(act)
        total_reward += R
        S = S_  # move to next state

        if vis and not done:
            if step + 1 % 100 == 0:
                print(f"[test_policy] Step {step + 1}")
            env.render()
            time.sleep(sleep_time)

        if done:
            print(f"[test_policy] Alcanza objetivo tras {step + 1} pasos")
            break
        elif step + 1 == max_steps:
            print(f"[test_policy] No alcanza objetivo tras {step + 1} pasos")

    return total_reward


def rl(env, epsilon, alpha, gamma, n_episodes,
       sleep_time=0.001, actions=9, vis=True, max_steps=1000,
       decay=0.99, min_epsilon=0.01, max_memory=5000, n_memory=500,
       batch_size=100, q_nnet=None, lr_decay=False):
    """Perform reinforcement learning."""
    if q_nnet is None:
        q_nnet = build_q_nnet(env.state_dim, actions, alpha, lr_decay)

    mem = deque(maxlen=max_memory)
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
            mem.append([S, act, R, S_, done])
            S = S_  # move to next state

            if (step + 1) % n_memory == 0:
                q_nnet = replay(q_nnet, gamma, mem, actions, batch_size)

            if vis and not done:
                if step + 1 % 100 == 0:
                    print(f"[Episodio {episode + 1}] Step {step + 1}")
                env.render()
                time.sleep(sleep_time)

            if done:
                steps_hist.append(step + 1)
                rewards_hist.append(total_reward + 1)
                q_nnet = replay(q_nnet, gamma, mem, actions,
                                (step + 1) % n_memory)  # TODO: ????
                print('Episodio %s: total steps = %s, total reward = %.2f' % (
                    episode + 1, step + 1, total_reward))
                break

            elif step + 1 == max_steps:
                print(
                    f"Episodio {episode + 1}: no alcanza tras {step + 1} pasos")

        if epsilon > min_epsilon:
            epsilon *= decay

    return q_nnet, episode + 1, steps_hist, rewards_hist


def keras_rl():
    env = myEnv(mode='hard')
    n_actions = env.action_space.n

    # Model
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(env.action_space.n))

    from rl.agents.dqn import DQNAgent
    from rl.memory import SequentialMemory
    from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

    memory = SequentialMemory(limit=5000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.0,
                                  nb_steps=50000)
    dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy, gamma=0.9)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=50000, visualize=False,
            verbose=2, nb_max_episode_steps=200)
    env2 = myEnv(mode='hard')
    dqn.test(env2, nb_episodes=100, visualize=True)


def main():
    np.random.seed(1)
    random.seed(1)

    env = myEnv(mode='hard')
    model = None  # load_model('q_nnet.h5')

    q_nnet, episodes, steps, rewards = rl(
        env,
        epsilon=0.25,
        alpha=0.01,
        lr_decay=True,
        gamma=0.95,
        n_episodes=1000,
        min_epsilon=0.01,
        vis=True,
        decay=0.99,
        max_steps=200,
        max_memory=100,
        n_memory=1,
        batch_size=16,
        q_nnet=model,
    )
    q_nnet.save('q_nnet.h5')

    # q_nnet = load_model('q_nnet.h5')
    env2 = myEnv(mode='hard')
    total_R = test_policy(env2, q_nnet, vis=True)
    print(f"Refuerzo total política: {total_R:.2f}")


if __name__ == '__main__':
    main()
