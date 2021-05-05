from collections import deque

import cv2
import gym
import numpy as np

from history import History
from model import Model
from trainer import Trainer


def preprocess(observation, observations):
    """
    Preprocess function for observation.
    """
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = observation[55:-55, 10:]
    observation = cv2.resize(observation, (84, 84))
    if observation.mean() > 200:
        observation = 255 - observation
    observations.append(observation)
    return np.stack(observations, axis=0)


def main():
    """
    Main procedure.
    """
    # Hyper-parameters.
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_step = 0.01
    batch_size = 64
    actions = [1, 2, 3, 7, 8]
    n_action = 5
    n_history = 50000
    n_episode = 5000
    n_observation = 4

    # Initalize instances.
    env = gym.make('Enduro-v0', frameskip=5)
    online = Model(n_action=n_action).cuda()
    target = Model(n_action=n_action).cuda()
    trainer = Trainer(online, target, gamma=gamma)
    history = History('s', 'a', 'r', 's*', 't', maxlen=n_history)

    for episode in range(n_episode):

        # Initialize the environment.
        observation, observations = env.reset(), deque(maxlen=n_observation)
        for _ in range(n_observation):
            state = preprocess(observation, observations)

        # Iterate until the episode is done.
        total, done = 0, False
        while not done:

            # Choose between exploration vs exploitation.
            if np.random.rand() <= epsilon:
                action = np.random.randint(n_action)
            else:
                action = online.predict(state)

            # Interact with the environment.
            observation, reward, done, _ = env.step(actions[action])
            consequence = preprocess(observation, observations)
            total += reward

            # Stack the experience tuple.
            history.append(state, action, reward, consequence, done)

            # Preserve the next state as a current state.
            state = consequence

        # Skip learning phase if it doesn't have enough history.
        if len(history) < n_history:
            continue

        # Checkpoint.
        trainer.save(f'checkpoint/{episode:04d}-{int(total):03d}.pt')

        # Epsilon schedule.
        epsilon = max(epsilon - epsilon_step, epsilon_min)

        # Mini-batch training.
        for replay in history.replay(batch_size):
            trainer.train(*replay)

        # Update target network.
        if episode % 5 == 0:
            trainer.update()


if __name__ == "__main__":
    main()
