import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from agent import myEnv
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay



class Memory():
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = []

    def append(self, item):
        self.memory.append(item)
        if self.len() > self.max_memory:
            del self.memory[0]

    def len(self):
        return len(self.memory)


def replay(q_nnet, gamma, mem, n_actions, batch_size=100):
    if batch_size <= 0:
        return q_nnet

    n_samples = min(batch_size, mem.len())
    minibatch = random.sample(mem.memory, n_samples)
    X_train = np.zeros((n_samples, minibatch[0][0].shape[0]))
    y_train = np.zeros((n_samples, 1, n_actions))

    for i, (S, A, R, S_, done) in enumerate(minibatch):
        if not done:
            # next state is not terminal
            q_target = R + gamma*q_nnet.predict(S_.reshape(1, -1)).max()
        else:
            q_target = R  # next state is terminal

        #target_vector = q_nnet.predict(S.reshape(1, -1))
        #target_vector[0][A] = q_target
        #q_nnet.fit(S.reshape(1,-1), target_vector, verbose=0) # update

        target_vector = q_nnet.predict(S.reshape(1, -1))
        target_vector[0][A] = q_target
        X_train[i] = S
        y_train[i] = target_vector

    q_nnet.train_on_batch(X_train, y_train)   # update

    return q_nnet


def build_q_nnet(n_state, n_actions, alpha, lr_decay, decay_steps):
    model = Sequential()
    model.add(Dense(200, input_shape=(n_state,),
                    bias_initializer="RandomNormal", activation="relu"))
    model.add(Dense(n_actions))

    if lr_decay:
        lr = ExponentialDecay(
            alpha,
            decay_steps=decay_steps,
            decay_rate=0.9,
            staircase=True)
    else:
        lr = alpha

    model.compile(loss='mse', optimizer=SGD(learning_rate=lr,
                                            momentum=0.9, nesterov=True))

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
       sleep_time=0.001, vis=True, max_steps=1000,
       decay=0.99, min_epsilon=0.01, max_memory=5000, n_memory=500,
       batch_size=100, q_nnet=None, lr_decay=False):
    """Perform reinforcement learning."""
    n_actions = env.action_space.n
    n_states = env.state_dim

    if q_nnet is None:
        decay_steps = n_episodes*(n_episodes/max_steps)
        q_nnet = build_q_nnet(n_states, n_actions,
                              alpha, lr_decay, decay_steps)

    mem = Memory(max_memory)
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
                q_nnet = replay(q_nnet, gamma, mem, n_actions, batch_size)

            if vis and not done:
                if (step + 1) % 100 == 0:
                    print(f"[Episodio {episode + 1}] step {step + 1}")
                env.render()
                time.sleep(sleep_time)

            if done:
                steps_hist.append(step + 1)
                rewards_hist.append(total_reward)
                print('[Episodio %s] total steps = %s, total reward = %.2f' % (
                    episode + 1, step + 1, total_reward))
                break

            elif step + 1 == max_steps:
                print(
                    f"[Episodio {episode + 1}] no alcanza tras {step + 1} pasos. "
                    f"Total reward = {total_reward:.2f}")

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


def test_policy(env, q_nnet, vis=True,
                sleep_time=0.001,
                max_steps=1000, verbose=True):
    S = env.reset()
    total_reward = 0

    for step in range(max_steps):
        act = np.argmax(q_nnet.predict(S.reshape(1, -1))[0])
        S_, R, done, _ = env.step(act)
        total_reward += R
        S = S_  # move to next state

        if vis and not done:
            if step + 1 % 100 == 0 and verbose:
                print(f"[test_policy] Step {step + 1}")
            env.render()
            time.sleep(sleep_time)

        if done:
            if verbose:
                print(f"[test_policy] Alcanza objetivo tras {step + 1} pasos")
            break
        elif step + 1 == max_steps and verbose:
            print(f"[test_policy] No alcanza objetivo tras {step + 1} pasos")

    return total_reward, done


def main():
    np.random.seed(1)
    random.seed(1)

    env = myEnv(mode='hard')
    model = None  # load_model('q_nnet.h5')

    q_nnet, episodes, steps, rewards = rl(
        env,
        epsilon=0.7,
        alpha=0.1,
        lr_decay=True,
        gamma=0.9,
        n_episodes=200,
        min_epsilon=0.1,
        vis=False,
        decay=0.99,
        max_steps=400,
        max_memory=5000,
        n_memory=200,
        batch_size=128,
        q_nnet=model,
    )
    q_nnet.save('q_nnet.h5')

    # Test 1 episode
    # q_nnet = load_model('q_nnet.h5')
    env2 = myEnv(mode='hard')
    total_R = test_policy(env2, q_nnet, vis=True)
    print(f"Refuerzo total política: {total_R:.2f}")

    # Test 100 episodes
    # q_nnet = load_model('q_nnet.h5')
    env2 = myEnv(mode='hard')
    n_success = 0
    mean_reward = 0.0

    for i in range(100):
        total_R, done = test_policy(env2, q_nnet, vis=False,
                                    verbose=False, max_steps=500)
        if done:
            n_success += 1
            mean_reward += total_R

    print(f"Número de éxitos: [{n_success}/100]")
    print(f"Recompensa media en los éxitos: {mean_reward/n_success:.4f}")


if __name__ == '__main__':
    main()
